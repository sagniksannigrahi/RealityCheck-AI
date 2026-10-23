"""Download a large file with parallel HTTP range requests (resumable). Usage: parallel_download.py URL OUT [N]."""

from __future__ import annotations

import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def size_of(url: str) -> int:
    if not url.startswith(("http://", "https://")):
        raise ValueError("only http(s) URLs are supported")
    req = urllib.request.Request(url, method="HEAD")  # noqa: S310
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed https/http dataset URLs
        return int(r.headers["Content-Length"])


def fetch(url: str, start: int, end: int, part: Path) -> None:
    have = part.stat().st_size if part.exists() else 0
    if have >= end - start + 1:
        return
    for _attempt in range(20):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})  # noqa: S310
            with urllib.request.urlopen(req, timeout=120) as r, part.open("ab") as f:  # noqa: S310
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            have = part.stat().st_size
            if have >= end - start + 1:
                return
        except Exception:  # noqa: BLE001 - retry on network errors
            have = part.stat().st_size if part.exists() else 0
    raise RuntimeError(f"failed range {start}-{end}")


def main() -> None:
    url, out = sys.argv[1], Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    total = size_of(url)
    step = total // n + 1
    parts = [(i * step, min(total - 1, (i + 1) * step - 1), out.with_suffix(out.suffix + f".part{i}")) for i in range(n)]
    with ThreadPoolExecutor(n) as ex:
        list(ex.map(lambda p: fetch(url, *p), parts))
    with out.open("wb") as f:
        for _, _, p in parts:
            f.write(p.read_bytes())
    for _, _, p in parts:
        p.unlink()
    assert out.stat().st_size == total, "size mismatch"
    print(f"downloaded {out} ({total} bytes)")


if __name__ == "__main__":
    main()
