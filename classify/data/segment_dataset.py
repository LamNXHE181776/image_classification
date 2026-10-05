"""PyTorch Dataset and LightningDataModule for folder-based segmentation.

Dataset layout expected by :class:`SegmentationDataModule`:: following the YOLOv8 convention:

    data_dir/
        img/
            train/
                img1.jpg
                img2.png
            val/
                img3.jpg
                img4.png
        mask/
            train/
                mask1.txt
                mask2.txt
            val/
                mask3.txt
                mask4.txt
        data.yaml
"""


from __future__ import annotations

import os
from typing import Tuple

import cv2
import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Dataset
import yaml

from classify.data.augmentation import Transformation
from classify.data.utils import create_dataset_dict, letterbox
from classify.utils import load_yaml


class SegmentationDataset(Dataset):
    """Folder-based image segmentation dataset.

    Args:
        dataset_dict: Mapping ``{image_path: mask_path}``.
        resize: Target square size for letterboxing.
        aug: Whether to apply training augmentation.
        label_to_index: Mapping ``{label: integer_class_id}``.
        color: Letterbox padding colour as ``(R, G, B)``.
    """

    def __init__(
            self,
            dataset_dict: dict[str, str],
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

        self.X: list[str] = [] # list of image file paths
        self.Y: list[str] = [] # list of corresponding mask file paths
        self.data_yaml = load_yaml(os.path.join(os.path.dirname(list(dataset_dict.keys())[0]), '..', 'data.yaml'))
        for image_path, mask_path in dataset_dict.items():
            self.X.append(image_path)
            self.Y.append(mask_path)


    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a pre-processed image tensor and its corresponding mask.

        Args:
            idx: Sample index.

        Returns:
            Tuple of ``(float32_chw_array, mask_array)``.
        """
        image = cv2.imread(self.X[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = letterbox(image, self.resize, color=self.color)

        mask_path = self.Y[idx]
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


        return image, mask


class SegmentationDataModule(L.LightningDataModule):
    """Lightning DataModule for segmentation tasks.

    Args:
        data_dir: Root directory with ``train/`` and ``val/`` sub-folders.
        resize: Square letterbox target size.
        batch_size: Mini-batch size for both train and val loaders.
        num_workers: Number of DataLoader worker processes.
        pad_color: Scalar grey value used for letterbox padding (0–255).
    """

    def __init__(
            self,
            data_dir: str,
            resize: int = 256,
            batch_size: int = 16,
            num_workers: int = 4,
            pad_color: int = 255,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.resize = resize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pad_color = (pad_color, pad_color, pad_color)