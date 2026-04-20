"""Load a collage layer image from a remote URL or local path."""

from __future__ import annotations

import io
import os
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

from PIL import Image


def load_layer_rgba(spec: str) -> Image.Image:
    """
    spec:
      - Absolute or relative filesystem path to PNG/JPEG (preferred for local work)
      - file:///path (three slashes on Unix)
      - http(s)://...
    """
    s = spec[:-1] if spec.endswith("/") else spec

    parsed = urlparse(s)
    if parsed.scheme == "file":
        path = unquote(parsed.path or "")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Layer image not found for file URL: {spec!r} -> {path!r}")
        return Image.open(path).convert("RGBA")

    if os.path.isfile(s):
        return Image.open(s).convert("RGBA")

    response = urlopen(s)
    data = response.read()
    return Image.open(io.BytesIO(data)).convert("RGBA")
