"""
Ollivier-Ricci curvature gate for poisoning defense on histopathology images.

This replaces the Bloch-sphere/quantum framing (bloch_gate.py) with a
principled graph-curvature signal, in the same spirit as the RicciGuard
intrusion-detection pipeline: build a k-NN graph over sample embeddings,
compute Ollivier-Ricci curvature on each edge via optimal transport between
neighbor distributions, and use curvature-weighted label agreement with a
sample's local neighborhood as the trust signal for loss reweighting.

Why this instead of Bloch-sphere geometry:
- Ollivier-Ricci curvature has a real graph-theoretic meaning (edges inside
  a tight cluster/community have high positive curvature; edges bridging
  two different regions of the embedding manifold have low or negative
  curvature). That property is exactly what you want for poisoning
  detection: a mislabeled sample sits geometrically inside a genuine
  same-*true*-class neighborhood (high-curvature edges) but disagrees with
  that neighborhood's consensus label.
- It's the same mechanism (Ollivier-Ricci curvature -> trust/feature
  weighting) already validated on tabular network-flow data in RicciGuard,
  applied here to CNN embedding space instead of a tabular feature graph —
  a genuine cross-domain transfer rather than a new ad hoc heuristic.
- No physics dressing (no Bloch vectors, purity, Bures metric) that isn't
  doing real work relative to a plain k-NN/curvature computation.

Curvature computation:
    For an edge (x, y) with neighbor distributions m_x, m_y (idleness alpha
    at the node itself, uniform mass over its k nearest neighbors):
        kappa(x, y) = 1 - W1(m_x, m_y) / d(x, y)
    W1 (1-Wasserstein / earth mover's distance) is approximated via a
    batched entropic-regularized Sinkhorn solver so the whole thing stays
    differentiable and runs on-device without an external OT dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RicciEmbedder(nn.Module):
    """Projects high-dimensional CNN embeddings down to a smaller space
    where the k-NN graph / curvature computation is cheap and stable.
    Architecturally mirrors BlochMapper's projection head, minus the
    tanh-to-a-ball normalization (there's no physical constraint to
    respect here — it's just a learned metric-embedding projection)."""

    def __init__(self, embed_dim=2048, hidden_dim=256, proj_dim=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, proj_dim),
        )

    def forward(self, x):
        return self.proj(x)


def sinkhorn_w1(cost, mu, nu, epsilon=0.1, n_iters=50):
    """
    Batched entropic-regularized OT (Sinkhorn) approximation of the
    1-Wasserstein transport cost between distributions mu and nu supported
    on the rows/cols of `cost`.

    Args:
        cost: (E, n, m) ground-distance cost matrices, one per edge
        mu:   (E, n) source distribution, rows sum to 1
        nu:   (E, m) target distribution, rows sum to 1
    Returns:
        (E,) approximate transport cost sum(P * cost) for each edge
    """
    # Normalize each edge's cost matrix by its own scale before exponentiating.
    # Without this, edges with very different absolute distances (e.g. a
    # within-cluster edge at scale ~0.1 vs. a cross-cluster edge at scale
    # ~30) blow up or underflow exp(-cost/epsilon) at a single fixed
    # epsilon, silently producing a degenerate (near-zero-cost) transport
    # plan instead of the true OT solution.
    scale = cost.detach().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    cost_norm = cost / scale
    K = torch.exp(-cost_norm / epsilon).clamp_min(1e-30)
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)
    for _ in range(n_iters):
        u = mu / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
        v = nu / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
    P = u.unsqueeze(-1) * K * v.unsqueeze(-2)
    return (P * cost).sum(dim=(1, 2))


class OllivierRicciGate(nn.Module):
    """
    Poisoning defense gate using Ollivier-Ricci curvature on a batch-level
    k-NN graph over projected embeddings.

    Mechanism:
    1. Build a k-NN graph (label-agnostic) over the batch's projected
       embeddings.
    2. Compute Ollivier-Ricci curvature on every (node, neighbor) edge via
       Sinkhorn-approximated W1 distance between the two nodes' local
       neighbor distributions.
    3. Curvature-weighted label agreement: does a sample's label agree with
       its geometrically well-connected (high-curvature) neighbors? Low
       agreement under high-curvature edges is the poisoning signal.

    Trust score: curvature-weighted neighbor label agreement, clamped to
    [0.1, 1.0] (same floor as the Bloch gate, to prevent zero-weight
    collapse of a class).
    """

    def __init__(self, embed_dim=2048, hidden_dim=256, proj_dim=32,
                 k_neighbors=5, alpha=0.5, sinkhorn_epsilon=0.1,
                 sinkhorn_iters=50, curvature_threshold=0.0,
                 gate_mode="soft"):
        super().__init__()
        self.embedder = RicciEmbedder(embed_dim, hidden_dim, proj_dim)
        self.k = k_neighbors
        self.alpha = alpha
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.curvature_threshold = curvature_threshold
        self.gate_mode = gate_mode

        # Auxiliary classifier on the projected embedding (mirrors
        # bloch_classifier in bloch_gate.py)
        self.ricci_classifier = nn.Linear(proj_dim, 2)

        # Learnable per-class trust bias (prevents minority-class collapse)
        self.trust_bias = nn.Parameter(torch.zeros(2))

    def _knn_graph(self, emb):
        """Returns pairwise dist (B,B), knn_idx (B,k), support_idx (B,k+1)."""
        B = emb.size(0)
        k = min(self.k, B - 1)
        dist = torch.cdist(emb, emb, p=2)
        dist_no_self = dist.clone()
        dist_no_self.fill_diagonal_(float('inf'))
        # Break exact ties deterministically by index: torch.topk's tie
        # handling on CUDA is not guaranteed stable/reproducible across
        # runs, and exact-tie distances do happen (e.g. duplicate/near-
        # duplicate embeddings early in training). A tiny index-dependent
        # offset makes neighbor selection a pure function of (emb, index)
        # regardless of hardware/kernel scheduling.
        tie_break = torch.arange(B, device=emb.device, dtype=dist.dtype).unsqueeze(0) * 1e-9
        knn_idx = torch.topk(dist_no_self + tie_break, k, largest=False).indices  # (B,k)
        self_idx = torch.arange(B, device=emb.device).unsqueeze(1)
        support_idx = torch.cat([self_idx, knn_idx], dim=1)  # (B,k+1)
        return dist, knn_idx, support_idx, k

    def ricci_curvature_edges(self, emb):
        """
        Computes Ollivier-Ricci curvature for every (node -> its k nearest
        neighbors) directed edge in the batch.

        Returns:
            src_idx, tgt_idx: (E,) edge endpoints
            kappa: (E,) curvature per edge, ordered so that kappa.view(B, k)
                gives a (node, neighbor) matrix directly (see forward() --
                this ordering is relied on there to avoid scatter_add_,
                which is not deterministic on CUDA)
        """
        B = emb.size(0)
        dist, knn_idx, support_idx, k = self._knn_graph(emb)

        if k == 0:
            empty = torch.zeros(0, dtype=torch.long, device=emb.device)
            return empty, empty, torch.zeros(0, device=emb.device)

        src_idx = torch.arange(B, device=emb.device).unsqueeze(1).expand(B, k).reshape(-1)
        tgt_idx = knn_idx.reshape(-1)
        E = src_idx.size(0)

        S_src = support_idx[src_idx]  # (E, k+1)
        S_tgt = support_idx[tgt_idx]  # (E, k+1)

        # cost[e, a, b] = dist[S_src[e,a], S_tgt[e,b]]
        cost = dist[S_src.unsqueeze(2).expand(-1, -1, k + 1),
                    S_tgt.unsqueeze(1).expand(-1, k + 1, -1)]

        rest_mass = (1.0 - self.alpha) / k
        mu = torch.full((E, k + 1), rest_mass, device=emb.device)
        mu[:, 0] = self.alpha
        nu = mu.clone()

        w1 = sinkhorn_w1(cost, mu, nu, epsilon=self.sinkhorn_epsilon,
                          n_iters=self.sinkhorn_iters)
        ground_dist = dist[src_idx, tgt_idx].clamp_min(1e-6)
        kappa = 1.0 - w1 / ground_dist
        return src_idx, tgt_idx, kappa

    def forward(self, embedding, labels=None, class_weights=None):
        """
        Args:
            embedding: (B, embed_dim) from ResNet-50
            labels: (B,) optional, for curvature-weighted label agreement
        Returns:
            gate_out: dict with trust_score, curvature, agreement, ricci_logits
        """
        proj = self.embedder(embedding)
        B = embedding.size(0)
        ricci_logits = self.ricci_classifier(proj)

        src_idx, tgt_idx, kappa = self.ricci_curvature_edges(proj)

        mean_curvature = torch.zeros(B, device=embedding.device)
        agreement = torch.full((B,), 0.5, device=embedding.device)

        if src_idx.numel() > 0:
            k = src_idx.numel() // B
            # edge weight: positively-curved (cluster-consistent) edges
            # count more than near-zero/negative (bridge) edges
            edge_weight = F.relu(kappa) + 0.01

            # By construction (see ricci_curvature_edges), edges are
            # grouped contiguously per source node in row-major order, so
            # a plain reshape+mean gives the same per-node aggregate as
            # scatter_add_ would -- without scatter_add_'s CUDA
            # non-determinism (atomic-accumulation order is not fixed
            # across runs, which was silently producing different trust
            # scores -- and therefore different final models -- from
            # identical seeds/configs).
            mean_curvature = kappa.view(B, k).mean(dim=1)

            if labels is not None:
                same_label = (labels[src_idx] == labels[tgt_idx]).float()
                w_sum = edge_weight.view(B, k).sum(dim=1)
                w_agree = (edge_weight * same_label).view(B, k).sum(dim=1)
                agreement = w_agree / w_sum.clamp_min(1e-6)

        # --- Trust score computation ---
        if labels is not None:
            base_trust = agreement
            bias = self.trust_bias[labels]
            trust_score = torch.sigmoid(base_trust * 4.0 - 2.0 + bias)  # spread agreement in [0,1] through a sigmoid
        else:
            # no label info (pure inference): fall back to normalized
            # curvature as a "how well-embedded / typical is this sample" signal
            trust_score = torch.sigmoid(mean_curvature)

        trust_score = trust_score.clamp(0.1, 1.0)

        if self.gate_mode == "soft":
            pass
        elif self.gate_mode == "hard":
            mask = (mean_curvature > self.curvature_threshold)
            if labels is not None:
                mask = mask & (agreement > 0.5)
            trust_score = mask.float().clamp_min(0.1)
        elif self.gate_mode == "curvature_only":
            trust_score = torch.sigmoid(mean_curvature).clamp(0.1, 1.0)

        return {
            "proj": proj,
            "curvature": mean_curvature,
            "agreement": agreement,
            "trust_score": trust_score,
            "ricci_logits": ricci_logits,
        }


class RicciSphereClassifier(nn.Module):
    """
    Full model: ResNet-50 backbone + Ollivier-Ricci gate + classifier.
    Drop-in replacement for BlochSphereClassifier — same forward/compute_loss
    contract so training_defense.py can select between defenses via a flag.
    """

    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False,
                 gate_config=None, gate_loss_weight=0.1):
        super().__init__()
        import torchvision.models as models
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.embed_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        gate_cfg = gate_config or {}
        self.gate = OllivierRicciGate(embed_dim=self.embed_dim, **gate_cfg)
        self.classifier_head = nn.Linear(self.embed_dim, num_classes)
        self.gate_loss_weight = gate_loss_weight
        self.class_weights = None

    def get_embedding(self, x):
        return self.backbone(x)

    def forward(self, x, labels=None, return_gate=False):
        emb = self.get_embedding(x)
        logits = self.classifier_head(emb)
        gate_out = self.gate(emb, labels)

        if return_gate:
            return logits, gate_out
        return logits

    def compute_loss(self, logits, labels, gate_out=None):
        """Standard CE loss reweighted by gate trust scores, plus an
        auxiliary classification loss on the projected embedding."""
        if gate_out is not None:
            trust = gate_out["trust_score"].detach()
            ce_loss = F.cross_entropy(logits, labels, weight=self.class_weights, reduction='none')
            ce_loss = (trust * ce_loss).mean()

            ricci_logits = gate_out["ricci_logits"]
            ricci_loss = F.cross_entropy(ricci_logits, labels, weight=self.class_weights)

            total_loss = ce_loss + self.gate_loss_weight * ricci_loss

            return total_loss, {
                "ce_loss": ce_loss.item(),
                "ricci_loss": ricci_loss.item(),
                "mean_trust": trust.mean().item(),
                "mean_curvature": gate_out["curvature"].mean().item(),
                "mean_agreement": gate_out["agreement"].mean().item(),
            }
        else:
            return F.cross_entropy(logits, labels, weight=self.class_weights), {}

    def set_class_weights(self, weights):
        self.class_weights = weights