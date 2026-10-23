"""Safe image decoding, cropping and encoding helpers."""

from __future__ import annotations

import hashlib
import io
import warnings

import numpy as np
from PIL import Image, ImageOps

from app.schemas.evidence import BoundingBox


class ImageDecodeError(ValueError):
    pass


def decode_image(data: bytes, *, max_pixels: int = 40_000_000, max_side: int = 8192) -> Image.Image:
    """Decode bytes into an RGB PIL image with decompression-bomb protection.

    The image is verified first (structure only), then re-opened and fully loaded.
    """
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(io.BytesIO(data))
            w, h = probe.size
            if w <= 0 or h <= 0 or w > max_side or h > max_side or w * h > max_pixels:
                raise ImageDecodeError(f"Image dimensions {w}x{h} exceed limits.")
            probe.verify()
            img = Image.open(io.BytesIO(data))
            img.load()
    except ImageDecodeError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageDecodeError("Image rejected: decompression bomb protection triggered.") from exc
    except Exception as exc:  # noqa: BLE001 - any decoder failure is a validation error
        raise ImageDecodeError(f"Could not decode image: {type(exc).__name__}") from exc
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_fingerprint(img: Image.Image) -> str:
    """Content hash of decoded pixels (stable across re-encoding of the same pixels)."""
    arr = np.asarray(img.convert("RGB"))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def crop(img: Image.Image, box: BoundingBox, pad: float = 0.05, min_side: int = 16) -> Image.Image:
    """Crop a normalized box with relative padding; never returns an empty image."""
    w, h = img.size
    bw, bh = box.width * w, box.height * h
    x1 = max(0, int(box.x1 * w - pad * bw))
    y1 = max(0, int(box.y1 * h - pad * bh))
    x2 = min(w, int(box.x2 * w + pad * bw))
    y2 = min(h, int(box.y2 * h + pad * bh))
    if x2 - x1 < min_side:
        cx = (x1 + x2) // 2
        x1, x2 = max(0, cx - min_side // 2), min(w, cx + min_side // 2)
    if y2 - y1 < min_side:
        cy = (y1 + y2) // 2
        y1, y2 = max(0, cy - min_side // 2), min(h, cy + min_side // 2)
    return img.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))


def encode_jpeg(img: Image.Image, quality: int = 85, max_side: int | None = None) -> bytes:
    if max_side and max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def image_quality(img: Image.Image) -> dict[str, float]:
    """Cheap image-quality indicators used to explain uncertainty (blur, darkness, low resolution)."""
    import cv2

    arr = np.asarray(img.convert("L"))
    small = arr
    if max(arr.shape) > 1024:
        scale = 1024 / max(arr.shape)
        small = cv2.resize(arr, (int(arr.shape[1] * scale), int(arr.shape[0] * scale)))
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
    brightness = float(small.mean())
    contrast = float(small.std())
    w, h = img.size
    return {
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "width": float(w),
        "height": float(h),
        "low_resolution": float(min(w, h) < 224),
        "blurry": float(sharpness < 40.0),
        "too_dark": float(brightness < 35.0),
        "too_bright": float(brightness > 230.0),
        "low_contrast": float(contrast < 15.0),
    }
