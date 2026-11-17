"""CNN-based image encoder (ResNet18, ImageNet pretrained).

The classifier head of ResNet18 is removed so the network behaves as a pure
feature extractor producing a 512-dimensional visual embedding.
"""
from __future__ import annotations

import threading
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image

import config


class ImageEncoderUnavailable(RuntimeError):
    pass


class ImageEncoder:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, image_size: int = config.IMAGE_SIZE):
        self.image_size = image_size
        self._model = None
        self._transform = None
        self._load_lock = threading.Lock()
        self.feature_dim = 512

    @classmethod
    def instance(cls) -> "ImageEncoder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torchvision
                from torchvision import transforms
                from torchvision.models import ResNet18_Weights, resnet18
            except ImportError as exc:  # pragma: no cover
                raise ImageEncoderUnavailable(
                    "torchvision is not installed. Run `pip install -r requirements.txt`."
                ) from exc

            try:
                weights = ResNet18_Weights.IMAGENET1K_V1
                net = resnet18(weights=weights)
            except Exception as exc:
                raise ImageEncoderUnavailable(
                    f"Could not load ResNet18 weights: {exc}. The first run needs "
                    "internet access to download them."
                ) from exc

            net.fc = torch.nn.Identity()  # (n, 512) pooled conv features
            net.eval()
            for param in net.parameters():
                param.requires_grad_(False)
            self._model = net
            self._torchvision_version = torchvision.__version__
            self._transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    @property
    def ready(self) -> bool:
        return self._model is not None

    def describe(self) -> dict:
        return {
            "model": "resnet18 (ImageNet IMAGENET1K_V1)",
            "architecture": "ResNet18 convolutional encoder, classifier head removed",
            "embedding_dim": self.feature_dim,
            "input_size": f"{self.image_size}x{self.image_size}",
            "normalisation": "ImageNet mean/std",
        }

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.load()
        if not images:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        batch = torch.stack([self._transform(im.convert("RGB")) for im in images])
        with torch.inference_mode():
            feats = self._model(batch)
        return feats.detach().cpu().numpy().astype(np.float32)
