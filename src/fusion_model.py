"""Multimodal fusion network and the two single-modality baselines.

Layout of the fusion network (Module 5 of the report):

    text  (768) --text_proj-->  (256) --\
                                        >-- gated concat (512) -- fusion MLP --> 64
    image (512) --image_proj--> (256) --/                                   |--> credibility head (2)
                                                                            |--> cross-modal consistency head (2)

The projectors push both modalities into one shared 256-d space, which is also
where cross-modal cosine similarity is measured. Modality gates are learned per
example, so the network can down-weight a modality that carries no signal
(e.g. the all-zero image vector used for text-only input).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_DIM = 768
IMAGE_DIM = 512
PROJECTION_DIM = 256
FUSED_DIM = PROJECTION_DIM * 4   # gated text, gated image, product, |difference|
HEAD_DIM = 64


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FusionNet(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.text_proj = _MLP(TEXT_DIM, 384, PROJECTION_DIM, dropout)
        self.image_proj = _MLP(IMAGE_DIM, 256, PROJECTION_DIM, dropout)
        self.gate = nn.Linear(PROJECTION_DIM * 2, 2)
        self.fusion = nn.Sequential(
            nn.Linear(FUSED_DIM, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, HEAD_DIM),
            nn.GELU(),
        )
        self.classifier = nn.Linear(HEAD_DIM, 2)      # 0 = credible, 1 = misleading
        self.consistency = nn.Linear(HEAD_DIM, 2)     # 0 = inconsistent, 1 = consistent

    def forward(self, text_feat, image_feat):
        t = F.normalize(self.text_proj(text_feat), p=2, dim=-1)
        i = F.normalize(self.image_proj(image_feat), p=2, dim=-1)
        gates = torch.softmax(self.gate(torch.cat([t, i], dim=-1)), dim=-1)
        # Explicit cross-modal interaction terms: the classifier sees agreement and
        # disagreement per dimension instead of having to infer it from a concatenation.
        fused = torch.cat([
            t * gates[:, 0:1],
            i * gates[:, 1:2],
            t * i,
            (t - i).abs(),
        ], dim=-1)
        head = self.fusion(fused)
        return (
            self.classifier(head),
            self.consistency(head),
            torch.sum(t * i, dim=-1),   # cosine similarity in the shared space
            gates,
        )


class TextOnlyNet(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(TEXT_DIM, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, HEAD_DIM), nn.GELU(),
        )
        self.classifier = nn.Linear(HEAD_DIM, 2)

    def forward(self, text_feat):
        return self.classifier(self.body(text_feat))


class ImageOnlyNet(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(IMAGE_DIM, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, HEAD_DIM), nn.GELU(),
        )
        self.classifier = nn.Linear(HEAD_DIM, 2)

    def forward(self, image_feat):
        return self.classifier(self.body(image_feat))


MODEL_REGISTRY = {
    "fusion": FusionNet,
    "text_only": TextOnlyNet,
    "image_only": ImageOnlyNet,
}
