"""PyTorch Dataset and LightningDataModule for folder-based classification.

Dataset layout expected by :class:`ClassificationDataModule`::

    data_dir/
        train/
            class_a/img1.jpg
            class_b/img2.png
        val/
            class_a/img3.jpg
            class_b/img4.png
"""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Dataset

from classify.data.augmentation import Transformation
from classify.data.utils import create_dataset_dict, letterbox


class ClassificationDataset(Dataset):
    """Folder-based image classification dataset.

    Args:
        dataset_dict: Mapping ``{label: [image_path, ...]}``.
        resize: Target square size for letterboxing.
        aug: Whether to apply training augmentation.
        label_to_index: Mapping ``{label: integer_class_id}``.
        color: Letterbox padding colour as ``(R, G, B)``.
    """

    def __init__(
        self,
        dataset_dict: dict[str, list[str]],
        resize: int,
        aug: bool,
        label_to_index: dict[str, int],
        color: tuple[int, int, int],
    ) -> None:
        self.resize = resize
        self.aug = aug
        self.color = color
        self.label_to_index = label_to_index
        self.transform = Transformation()

        self.X: list[str] = [] # List of image file paths
        self.Y: list[str] = [] # List of corresponding class labels
        for label, paths in dataset_dict.items():
            self.X.extend(paths)
            self.Y.extend([label] * len(paths))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Return a pre-processed image tensor and its integer class label.

        Args:
            idx: Sample index.

        Returns:
            Tuple of ``(float32_chw_array, class_index)``.
        """
        image = cv2.imread(self.X[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = letterbox(image, self.resize, color=self.color)

        if self.aug:
            image = self.transform(image)

        image = (image / 255.0).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        label_idx = self.label_to_index[self.Y[idx]]
        return image, label_idx


class ClassificationDataModule(L.LightningDataModule):
    """Lightning DataModule wrapping :class:`ClassificationDataset`.

    Args:
        data_dir: Root directory with ``train/`` and ``val/`` sub-folders.
        resize: Square letterbox target size.
        batch_size: Mini-batch size for both train and val loaders.
        apply_aug: Enable augmentation on the training split.
        num_workers: Number of DataLoader worker processes.
        pad_color: Scalar grey value used for letterbox padding (0–255).
    """

    def __init__(
        self,
        data_dir: str,
        resize: int = 256,
        batch_size: int = 16,
        apply_aug: bool = False,
        num_workers: int = 4,
        pad_color: int = 255,
        use_class_weights: bool = False,
    ) -> None:
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.data_dir = data_dir
        self.resize = resize
        self.batch_size = batch_size
        self.apply_aug = apply_aug
        self.num_workers = num_workers
        self.pad_color = (pad_color, pad_color, pad_color)
        self.use_class_weights = use_class_weights
        self.class_weights: np.ndarray | None = None  # To be set in setup()

    def setup(self, stage: str | None = None) -> None:
        """Build train and val datasets.

        Args:
            stage: Lightning stage string (ignored; both splits are always set up).

        Raises:
            ValueError: If either split contains zero samples.
        """
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")

        train_dict, label_to_index = create_dataset_dict(train_dir)
        val_dict, _ = create_dataset_dict(val_dir)


        if self.use_class_weights:
            class_len = {lbl: len(paths) for lbl, paths in train_dict.items()}
            total = sum(class_len.values())

            class_weights = total / np.array(list(class_len.values())) * len(class_len)
            class_weights = class_weights / class_weights.sum()

            self.class_weights = class_weights.astype(np.float32)

        if not train_dict or all(len(v) == 0 for v in train_dict.values()):
            raise ValueError(f"No training images found under '{train_dir}'.")
        if not val_dict or all(len(v) == 0 for v in val_dict.values()):
            raise ValueError(f"No validation images found under '{val_dir}'.")

        common_kwargs = dict(
            resize=self.resize,
            label_to_index=label_to_index,
            color=self.pad_color,
        )
        self.train_ds = ClassificationDataset(
            dataset_dict=train_dict, aug=self.apply_aug, **common_kwargs
        )
        self.val_ds = ClassificationDataset(
            dataset_dict=val_dict, aug=False, **common_kwargs
        )

        return train_dict, val_dict

    def _make_loader(self, dataset: ClassificationDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False)

