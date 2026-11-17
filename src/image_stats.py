"""Low-level visual statistics and heuristic artefact indicators.

These numbers describe *properties of the pixels*. They are not a forensic
manipulation detector: no such model is implemented in this project, and the
UI labels them accordingly.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _jpeg_quality(image: Image.Image) -> int | None:
    try:
        q = getattr(image, "quantization", None)
        if not q:
            return None
        table = q.get(0) if isinstance(q, dict) else None
        if not table:
            return None
        sums = float(sum(table))
        if sums <= 0:
            return None
        # Rough inverse mapping of the standard luminance table sum to quality.
        quality = int(round(100 - (sums - 400) / 45.0))
        return max(1, min(100, quality))
    except Exception:
        return None


def analyse(image: Image.Image, raw_bytes: bytes | None = None) -> dict:
    arr = _to_array(image)
    h, w = arr.shape[:2]
    grey = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    # Sharpness: variance of the Laplacian (4-neighbour kernel).
    lap = (-4 * grey[1:-1, 1:-1] + grey[:-2, 1:-1] + grey[2:, 1:-1]
           + grey[1:-1, :-2] + grey[1:-1, 2:])
    sharpness = float(lap.var())

    # Edge density via a simple Sobel magnitude.
    gx = np.zeros_like(grey)
    gy = np.zeros_like(grey)
    gx[:, :-2] = grey[:, 2:] - grey[:, :-2]
    gy[:-2, :] = grey[2:, :] - grey[:-2, :]
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    edge_density = float((magnitude > 0.22).mean())

    # Hasler-Suesstrunk colourfulness.
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colourfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                          + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    # Horizontal band occupying the bottom ~22% (caption overlays live there).
    band = magnitude[int(h * 0.78):, :]
    bottom_band_edges = float((band > 0.22).mean()) if band.size else 0.0
    overlay_hint = bottom_band_edges > max(0.06, edge_density * 1.6)

    # Blockiness: mean |difference| across 8x8 block boundaries vs interior.
    blk_w = grey[:, 7::8][:, :].shape
    boundary = np.abs(np.diff(grey, axis=1))[:, 7::8].mean() if grey.shape[1] > 9 else 0.0
    interior = np.abs(np.diff(grey, axis=1)).mean() if grey.shape[1] > 2 else 0.0
    blockiness = float(boundary / (interior + 1e-6))

    jpeg_quality = _jpeg_quality(image)
    flags = []
    if overlay_hint:
        flags.append({"name": "Caption/overlay band", "value": round(bottom_band_edges, 3),
                      "severity": "medium",
                      "detail": "high edge density concentrated in the lower band, the "
                                "pattern produced by burned-in caption text"})
    if jpeg_quality is not None and jpeg_quality <= 40:
        flags.append({"name": "Heavy re-compression", "value": jpeg_quality,
                      "severity": "medium",
                      "detail": "coarse JPEG quantisation typical of an image that has been "
                                "re-saved several times"})
    if blockiness > 1.6:
        flags.append({"name": "Block discontinuities", "value": round(blockiness, 2),
                      "severity": "low",
                      "detail": "8x8 block boundaries stand out from the interior, consistent "
                                "with a re-encoded or spliced region"})
    if sharpness < 0.0009:
        flags.append({"name": "Low detail", "value": round(sharpness, 6),
                      "severity": "low",
                      "detail": "very low high-frequency energy; the image may be heavily "
                                "blurred or downscaled"})

    return {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": round(w / max(1, h), 3),
        "mode": image.mode,
        "format": image.format or "PNG",
        "brightness": round(float(grey.mean()), 4),
        "contrast": round(float(grey.std()), 4),
        "saturation": round(float((arr.max(axis=2) - arr.min(axis=2)).mean()), 4),
        "sharpness": round(sharpness, 6),
        "edge_density": round(edge_density, 4),
        "colourfulness": round(colourfulness, 4),
        "jpeg_quality_estimate": jpeg_quality,
        "blockiness": round(blockiness, 3),
        "bytes": len(raw_bytes) if raw_bytes else None,
        "artefact_flags": flags,
        "forensics_available": False,
    }
