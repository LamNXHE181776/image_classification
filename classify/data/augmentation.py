"""Albumentations augmentation pipeline for image classification.

The pipeline is intentionally conservative (all ops at p=0.02) and mirrors
the production augmentation used across AICycle classification tasks.

To customise, pass an ``aug_config`` dict when building a
:class:`~classify.data.dataset.ClassificationDataModule`.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
import numpy as np
from numpy.typing import NDArray


def build_augmentation_pipeline(config: dict[str, Any] | None = None) -> A.Compose:
    """Construct an Albumentations augmentation pipeline.

    Args:
        config: Reserved for future per-task configuration.  Currently unused;
            pass ``None`` to get the default production preset.

    Returns:
        An :class:`albumentations.Compose` pipeline.
    """
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.06,
                scale_limit=0.05,
                rotate_limit=15,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.2,
            ),

            A.Perspective(
                scale=(0.05, 0.1),
                keep_size=True,
                p = 0.2
            ),

            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=0.4, contrast_limit=0.1),
                    A.CLAHE(clip_limit=1.5, tile_grid_size=(4, 4), p=0.3),
                    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1), p=0.3),
                ],
                p=0.4,
            ),

            A.OneOf(
                [
                    A.ImageCompression(
                        quality_range=(75,95),
                        compression_type="jpeg",
                        p=0.2,
                    ),
                    A.ImageCompression(
                        quality_range=(75, 95),
                        compression_type="webp",
                        p=0.2,
                    )
                ],
                p = 0.3
            ),
            A.GaussNoise(p=0.2),
        ]
    )


class Transformation:
    """Callable wrapper around an Albumentations pipeline.

    Args:
        config: Optional config dict forwarded to
            :func:`build_augmentation_pipeline`.

    Example::

        transform = Transformation()
        augmented = transform(image=my_rgb_array)  # numpy HxWx3 uint8
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.pipeline = build_augmentation_pipeline(config)

    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply the augmentation pipeline to a single image.

        Args:
            image: RGB numpy array of shape ``(H, W, 3)`` with dtype uint8.

        Returns:
            Augmented RGB numpy array with the same dtype and shape.
        """
        return self.pipeline(image=np.asarray(image))["image"]
