from __future__ import annotations


def register_heif_if_available() -> None:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return
    register_heif_opener()
