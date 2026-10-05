"""Dataset utilities — letterboxing and folder-based label discovery."""

from __future__ import annotations

import os
from glob import glob
from typing import List, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def letterbox(
    image: np.ndarray,
    resize: int,
    color: Union[Tuple[int, int, int], List[int]] = (0, 0, 0),
) -> NDArray[np.uint8]:
    """Resize an image to a square canvas using letterboxing (aspect-safe).

    The longest edge is scaled to *resize*; the shorter edge is padded with
    *color* on both sides symmetrically.

    Args:
        image: BGR or RGB numpy array of shape ``(H, W, 3)``.
        resize: Target square side length in pixels.
        color: Padding colour as an ``(R, G, B)`` tuple (default: black).

    Returns:
        Letterboxed image of shape ``(resize, resize, 3)`` as uint8.
    """
    h, w = image.shape[:2]
    if h >= w:
        h_new = resize
        w_new = int(w / h * resize)
    else:
        w_new = resize
        h_new = int(h / w * resize)

    image = cv2.resize(image, (w_new, h_new))

    top = (resize - h_new) // 2
    bottom = resize - h_new - top
    left = (resize - w_new) // 2
    right = resize - w_new - left

    return cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )


def get_dataset_classes(data_dir: str) -> List[str]:
    """Return a sorted list of class names found in ``<data_dir>/train/``.

    Each sub-folder directly inside ``<data_dir>/train/`` is treated as one
    class, matching the convention of :func:`create_dataset_dict`.

    Args:
        data_dir: Dataset root that contains a ``train/`` sub-folder.

    Returns:
        Sorted list of class-folder names.

    Raises:
        ValueError: If ``<data_dir>/train/`` does not exist or is empty.
    """
    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        raise ValueError(
            f"Cannot detect classes: '{train_dir}' does not exist."
        )
    classes = sorted(
        os.path.basename(p)
        for p in glob(os.path.join(train_dir, "*"))
        if os.path.isdir(p)
    )
    if not classes:
        raise ValueError(
            f"Cannot detect classes: no sub-folders found in '{train_dir}'."
        )
    return classes


def auto_num_classes(data_dir: str) -> int:
    """Count the number of classes from the ``train/`` split of a dataset.

    Thin wrapper around :func:`get_dataset_classes` that returns the count.

    Args:
        data_dir: Dataset root that contains a ``train/`` sub-folder.

    Returns:
        Number of class folders found.
    """
    return len(get_dataset_classes(data_dir))


def create_dataset_dict(data_dir: str) -> Tuple[dict[str, list[str]], dict[str, int]]:
    """Scan a folder-based classification dataset and return metadata.

    Expected layout::

        data_dir/
            class_a/image1.jpg
            class_a/image2.png
            class_b/image1.jpg
            ...

    Args:
        data_dir: Root directory containing one sub-folder per class.

    Returns:
        A two-tuple of:

        * ``dataset_dict`` — ``{label: [abs_image_path, ...]}``
        * ``label_to_index`` — ``{label: integer_index}`` sorted alphabetically.
    """
    labels: list[str] = sorted(
        os.path.basename(p)
        for p in glob(os.path.join(data_dir, "*"))
        if os.path.isdir(p)
    )
    label_to_index: dict[str, int] = {lbl: i for i, lbl in enumerate(labels)}
    dataset_dict: dict[str, list[str]] = {}

    for label in labels:
        dataset_dict[label] = [
            p
            for p in glob(os.path.join(data_dir, label, "*"))
            if os.path.splitext(p)[-1].lower() in _SUPPORTED_EXTENSIONS
        ]

    return dataset_dict, label_to_index
