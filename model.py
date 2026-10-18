"""
Baseline backbone: ImageNet-pretrained ResNet-50 -> global pool -> embedding
-> binary classifier head. This matches the "Baseline Backbone (CNN)" /
"ResNet-50 (maybe)" box in the architecture sketch, minus the geometric
transform / gate, which baselines 1 & 2 deliberately omit.

get_embedding() is exposed separately so the Bloch-sphere/curvature gate
(added later) can hook into the 2048-d embedding vector without changing
this file.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        embed_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()  # strip original classifier -> raw embedding
        self.backbone = backbone
        self.embed_dim = embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classifier_head = nn.Linear(embed_dim, num_classes)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding(x)
        logits = self.classifier_head(emb)
        return logits