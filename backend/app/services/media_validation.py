"""Upload validation: type sniffing from magic bytes, size limits and safe decoding.

The client-supplied filename and Content-Type are never trusted for type decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import Settings
from app.core.imaging import ImageDecodeError, decode_image


class MediaValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SniffResult:
    media_type: str  # image | video
    mime: str
    ext: str


def sniff(data: bytes) -> SniffResult:
    head = data[:64]
    if head.startswith(b"\xff\xd8\xff"):
        return SniffResult("image", "image/jpeg", "jpg")
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return SniffResult("image", "image/png", "png")
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return SniffResult("image", "image/webp", "webp")
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return SniffResult("image", "image/gif", "gif")
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"qt  ",):
            return SniffResult("video", "video/quicktime", "mov")
        if brand[:3] in (b"hei", b"mif", b"avi", b"msf"):  # HEIC/AVIF still images are not supported
            raise MediaValidationError("unsupported_media_type", "HEIC/AVIF images are not supported; upload JPEG or PNG.")
        return SniffResult("video", "video/mp4", "mp4")
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return SniffResult("video", "video/webm", "webm")
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return SniffResult("video", "video/x-msvideo", "avi")
    raise MediaValidationError("unsupported_media_type",
                               "Unsupported or unrecognized file format. Supported: JPEG, PNG, WebP, GIF, MP4, MOV, WebM, AVI.")


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]")


def sanitize_filename(name: str | None) -> str | None:
    if not name:
        return None
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    base = _SAFE_NAME.sub("_", base).strip(" .")[:120]
    return base or None


def validate_upload(data: bytes, settings: Settings) -> SniffResult:
    if not data:
        raise MediaValidationError("empty_file", "The uploaded file is empty.")
    kind = sniff(data)
    limit = settings.max_image_bytes if kind.media_type == "image" else settings.max_video_bytes
    if len(data) > limit:
        raise MediaValidationError("file_too_large",
                                   f"{kind.media_type.capitalize()} exceeds the {limit // (1024 * 1024)} MB limit.")
    return kind


def decode_validated_image(data: bytes, settings: Settings):
    try:
        return decode_image(data, max_pixels=settings.max_image_pixels, max_side=settings.max_image_side)
    except ImageDecodeError as exc:
        raise MediaValidationError("invalid_image", str(exc)) from exc
