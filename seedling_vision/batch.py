from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from seedling_core.schemas import DetectionResultV1, InferenceContext

from .adapters import DetectorAdapter


def batch_predict(
    detector: DetectorAdapter,
    images: Iterable[str | Path],
    context_factory: Callable[[str | Path], InferenceContext] | None = None,
    progress: Callable[[int, str | Path], None] | None = None,
) -> list[DetectionResultV1]:
    results = []
    for index, image in enumerate(images, 1):
        if progress:
            progress(index, image)
        context = context_factory(image) if context_factory else InferenceContext(image_id=Path(image).name)
        results.append(detector.predict(image, context=context))
    return results
