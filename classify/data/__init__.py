"""Data pipeline for image classification.

Public API::

    from classify.data import ClassificationDataModule, ClassificationDataset
    from classify.data import build_augmentation_pipeline
    from classify.data.utils import letterbox, create_dataset_dict
"""

from classify.data.augmentation import Transformation, build_augmentation_pipeline
from classify.data.dataset import ClassificationDataset, ClassificationDataModule
from classify.data.utils import letterbox, create_dataset_dict, auto_num_classes, get_dataset_classes

__all__ = [
    "Transformation",
    "build_augmentation_pipeline",
    "ClassificationDataset",
    "ClassificationDataModule",
    "letterbox",
    "create_dataset_dict",
    "auto_num_classes",
    "get_dataset_classes",
]
