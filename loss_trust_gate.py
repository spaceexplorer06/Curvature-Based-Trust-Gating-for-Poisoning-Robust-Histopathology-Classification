"""
Small-loss trust gate: a simple, non-geometric ablation baseline.

This is the "small-loss trick" (co-teaching / MentorNet family): samples
with unusually high per-sample loss relative to their batch are down-
weighted, on the assumption that mislabeled samples are harder for the
model to fit consistently. No embedding graph, no curvature, no learned
gate parameters beyond the classifier head itself.

Exists purely as a control: if the Ollivier-Ricci gate doesn't clearly beat
this, the curvature machinery isn't earning its complexity for a journal
submission.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallLossGate(nn.Module):
    """
    Computes a per-sample trust score from the batch-relative percentile of
    each sample's cross-entropy loss. No neighbor graph, no embeddings used
    beyond the logits/labels already produced by the classifier.
    """

    def __init__(self, keep_fraction=0.8, gate_mode="soft"):
        super().__init__()
        # keep_fraction: assumed clean fraction of any batch (co-teaching
        # default is often tied to an estimated poison rate; 0.8 is a
        # reasonable fixed default independent of the true poison rate,
        # which a real defense shouldn't get to see).
        self.keep_fraction = keep_fraction
        self.gate_mode = gate_mode

    def forward(self, logits, labels):
        per_sample_loss = F.cross_entropy(logits, labels, reduction='none')
        B = per_sample_loss.size(0)

        # rank 0 = lowest loss (most trustworthy) ... rank B-1 = highest loss
        order = torch.argsort(per_sample_loss)
        rank = torch.empty(B, device=logits.device)
        rank[order] = torch.arange(B, device=logits.device, dtype=torch.float32)
        percentile = rank / max(B - 1, 1)  # 0 = lowest loss, 1 = highest loss

        if self.gate_mode == "hard":
            keep_n = max(1, int(round(self.keep_fraction * B)))
            trust_score = (rank < keep_n).float().clamp_min(0.1)
        else:
            # soft: smoothly down-weight high-loss (high-percentile) samples
            trust_score = (1.0 - percentile).clamp(0.1, 1.0)

        return {
            "per_sample_loss": per_sample_loss,
            "loss_percentile": percentile,
            "trust_score": trust_score,
        }


class SmallLossClassifier(nn.Module):
    """
    Full model: ResNet-50 backbone + small-loss trust gate. Same
    forward/compute_loss contract as BlochSphereClassifier / RicciSphereClassifier.
    """

    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False,
                 gate_config=None, gate_loss_weight=0.0):
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
        self.gate = SmallLossGate(**gate_cfg)
        self.classifier_head = nn.Linear(self.embed_dim, num_classes)
        self.class_weights = None

    def get_embedding(self, x):
        return self.backbone(x)

    def forward(self, x, labels=None, return_gate=False):
        emb = self.get_embedding(x)
        logits = self.classifier_head(emb)
        gate_out = self.gate(logits, labels) if labels is not None else {
            "trust_score": torch.ones(x.size(0), device=x.device)
        }
        if return_gate:
            return logits, gate_out
        return logits

    def compute_loss(self, logits, labels, gate_out=None):
        if gate_out is not None and "per_sample_loss" in gate_out:
            trust = gate_out["trust_score"].detach()
            ce_loss = (trust * gate_out["per_sample_loss"]).mean()
            return ce_loss, {
                "ce_loss": ce_loss.item(),
                "mean_trust": trust.mean().item(),
                "mean_loss_percentile": gate_out["loss_percentile"].mean().item(),
            }
        else:
            return F.cross_entropy(logits, labels, weight=self.class_weights), {}

    def set_class_weights(self, weights):
        self.class_weights = weights