"""
Bloch-sphere Riemannian gate for poisoning defense on histopathology images.

Maps CNN embeddings to Bloch vectors (quantum two-state density matrices) and
uses geometric features (purity, local curvature) to detect poisoned samples.

For Oxford Bioinformatics:
- First work combining Bloch-sphere representation with poisoning defense
- on histopathology image embeddings (BreaKHis dataset)
- Physics-motivated: purity = state uncertainty, curvature = manifold anomaly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BlochMapper(nn.Module):
    """
    Maps high-dimensional CNN embeddings to 3D Bloch vectors.

    The Bloch vector r satisfies ||r|| <= 1, representing a qubit density matrix:
        rho = 1/2 * (I + r · sigma)
    where sigma = (sigma_x, sigma_y, sigma_z) are Pauli matrices.

    Purity Tr(rho^2) = 1/2 * (1 + ||r||^2) in [0.5, 1.0]
    """
    def __init__(self, embed_dim=2048, hidden_dim=256):
        super().__init__()
        self.mapper = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        # Learnable temperature for soft-normalization
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        Args:
            x: (B, embed_dim) embedding vectors
        Returns:
            r: (B, 3) Bloch vectors with ||r|| <= 1
        """
        r = self.mapper(x)
        # Soft-normalize: r_hat = tanh(||r||/T) * r/||r||
        # Ensures ||r_hat|| in [0, 1], with T controlling boundary reachability
        norm = torch.norm(r, dim=1, keepdim=True).clamp_min(1e-6)
        scale = torch.tanh(norm / self.temp.clamp_min(0.1))
        r_bloch = scale * (r / norm)
        return r_bloch

    def purity(self, r):
        """Purity Tr(rho^2) = 1/2 * (1 + ||r||^2). Range: [0.5, 1.0]."""
        return 0.5 * (1.0 + (r ** 2).sum(dim=1))

    def von_neumann_entropy(self, r):
        """
        S(rho) = -Tr(rho log rho) = -lambda_+ log lambda_+ - lambda_- log lambda_-
        where lambda_± = 1/2 * (1 ± ||r||)
        """
        norm_r = torch.norm(r, dim=1).clamp_min(1e-6)
        lam_plus = 0.5 * (1 + norm_r)
        lam_minus = 0.5 * (1 - norm_r)
        eps = 1e-8
        return -(lam_plus * torch.log(lam_plus + eps) + lam_minus * torch.log(lam_minus + eps))


class RiemannianGate(nn.Module):
    """
    Poisoning defense gate using Bloch-sphere geometric features.

    Two mechanisms:
    1. Purity gating: Low purity (high entropy) indicates ambiguous/suspicious samples
    2. Curvature gating: High local geodesic curvature relative to same-class neighbors
       flags poisoned label contradictions ("kinks" in the data manifold)

    Trust score: tau = purity_score * curvature_score in [0, 1]
    """
    def __init__(self, embed_dim=2048, hidden_dim=256, k_neighbors=5,
                 curvature_threshold=0.3, entropy_threshold=0.3, gate_mode="soft"):
        super().__init__()
        self.bloch_mapper = BlochMapper(embed_dim, hidden_dim)
        self.k = k_neighbors
        self.curvature_threshold = curvature_threshold
        self.entropy_threshold = entropy_threshold
        self.gate_mode = gate_mode

        # Auxiliary classifier on Bloch vectors
        self.bloch_classifier = nn.Linear(3, 2)

        # Learnable per-class trust bias (prevents minority class collapse)
        self.trust_bias = nn.Parameter(torch.zeros(2))

    def geodesic_distance(self, r1, r2):
        """
        Bures metric distance between two Bloch vectors.
        d_Bures^2 = 2 * (1 - sqrt(Fidelity))
        where Fidelity = 1/2 * (1 + r1·r2 + sqrt((1-||r1||^2)(1-||r2||^2)))
        """
        dot = (r1 * r2).sum(dim=-1)
        norm1_sq = (r1 ** 2).sum(dim=-1)
        norm2_sq = (r2 ** 2).sum(dim=-1)
        fidelity = 0.5 * (1 + dot + torch.sqrt((1 - norm1_sq).clamp_min(0) * (1 - norm2_sq).clamp_min(0)))
        fidelity = fidelity.clamp(0, 1)
        return torch.sqrt(2 * (1 - torch.sqrt(fidelity)))

    def local_curvature(self, r, labels):
        """
        Estimate local extrinsic curvature at each point.

        For each sample, compare mean distance to k same-class neighbors vs
        k different-class neighbors. High ratio indicates the sample is an
        outlier within its class — characteristic of poisoned labels.

        Poisoned samples: same-class neighbors are far (contradictory labels),
        creating high local curvature.
        """
        B = r.size(0)
        dists = torch.cdist(r, r, p=2)
        dists.fill_diagonal_(float('inf'))
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff_class = ~same_class

        curvature = torch.zeros(B, device=r.device)
        for i in range(B):
            same_mask = same_class[i]
            same_dists = dists[i].clone()
            same_dists[~same_mask] = float('inf')

            diff_mask = diff_class[i]
            diff_dists = dists[i].clone()
            diff_dists[~diff_mask] = float('inf')

            k_same = min(self.k, same_mask.sum().item())
            k_diff = min(self.k, diff_mask.sum().item())

            same_mean = torch.topk(same_dists, k_same, largest=False)[0].mean() if k_same > 0 else torch.tensor(0.0, device=r.device)
            diff_mean = torch.topk(diff_dists, k_diff, largest=False)[0].mean() if k_diff > 0 else torch.tensor(1.0, device=r.device)

            curvature[i] = same_mean / (diff_mean + 1e-6)

        return curvature

    def forward(self, embedding, labels=None, class_weights=None):
        """
        Args:
            embedding: (B, embed_dim) from ResNet-50
            labels: (B,) optional, for supervised curvature computation
        Returns:
            gate_out: dict with trust_score, purity, curvature, bloch_logits
        """
        r = self.bloch_mapper(embedding)

        purity = self.bloch_mapper.purity(r)
        entropy = self.bloch_mapper.von_neumann_entropy(r)
        bloch_logits = self.bloch_classifier(r)

        # Curvature-based anomaly detection
        if labels is not None and self.training:
            curvature = self.local_curvature(r, labels)
        else:
            curvature = torch.zeros(embedding.size(0), device=embedding.device)

        # --- Trust score computation ---

        # 1. Purity gating: low purity = high entropy = suspicious
        max_entropy = np.log(2)
        purity_score = 1.0 - (entropy / max_entropy).clamp(0, 1)

        # 2. Curvature gating: high curvature = poisoned
        curvature_score = torch.exp(-2.0 * curvature)

        # Combined trust score with per-class bias
        if labels is not None:
            bias = self.trust_bias[labels]
            trust_score = torch.sigmoid(purity_score * curvature_score + bias)
        else:
            trust_score = purity_score * curvature_score

        # Clip: minimum trust to prevent zero-weight collapse
        trust_score = trust_score.clamp(0.1, 1.0)

        if self.gate_mode == "soft":
            pass
        elif self.gate_mode == "hard":
            mask = (curvature < self.curvature_threshold) & (purity_score > 0.3)
            trust_score = mask.float()
        elif self.gate_mode == "curvature_only":
            trust_score = curvature_score.clamp(0.1, 1.0)

        return {
            "bloch_vec": r,
            "purity": purity,
            "entropy": entropy,
            "curvature": curvature,
            "trust_score": trust_score,
            "bloch_logits": bloch_logits,
        }


class BlochSphereClassifier(nn.Module):
    """
    Full model: ResNet-50 backbone + Bloch-sphere gate + classifier.

    During training, the gate reweights loss by trust_score.
    During inference, the gate flags suspicious samples.
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
        self.gate = RiemannianGate(embed_dim=self.embed_dim, **gate_cfg)
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
        """
        Standard CE loss, reweighted by gate trust scores.
        Also adds auxiliary Bloch-sphere classification loss.
        """
        if gate_out is not None:
            trust = gate_out["trust_score"].detach()
            ce_loss = F.cross_entropy(logits, labels, weight=self.class_weights, reduction='none')
            ce_loss = (trust * ce_loss).mean()

            # Auxiliary Bloch classifier loss
            bloch_logits = gate_out["bloch_logits"]
            bloch_loss = F.cross_entropy(bloch_logits, labels, weight=self.class_weights)

            total_loss = ce_loss + self.gate_loss_weight * bloch_loss

            return total_loss, {
                "ce_loss": ce_loss.item(),
                "bloch_loss": bloch_loss.item(),
                "mean_trust": trust.mean().item(),
                "mean_purity": gate_out["purity"].mean().item(),
                "mean_curvature": gate_out["curvature"].mean().item(),
            }
        else:
            return F.cross_entropy(logits, labels, weight=self.class_weights), {}

    def set_class_weights(self, weights):
        self.class_weights = weights