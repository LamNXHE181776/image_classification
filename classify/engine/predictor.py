"""Inference engine for trained classification models.

Usage (programmatic)::

    from classify.engine.predictor import Predictor

    predictor = Predictor(
        model_name="efficientnet_v2_s",
        num_classes=4,
        resize=512,
        pad_color=255,
        ckpt_path="best.ckpt",
        device="cuda",
    )

    # Single image (numpy RGB array)
    cls_idx, score = predictor.predict(image, is_bgr=False)

    # Batch over a folder or file list via CLI helper
    predictor.run(source="/path/to/images/")
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from classify.models.lightning_module import ImageClassificationModule
from classify.data.utils import letterbox
from classify.utils.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})


class Predictor:
    """Loads a trained checkpoint and runs classification inference.

    Args:
        model_name: Architecture name matching the checkpoint.
        num_classes: Number of classes the model was trained with.
        resize: Square input size used during training.
        pad_color: Letterbox padding colour (scalar 0–255).
        ckpt_path: Path to the ``.ckpt`` file.
        device: ``"cuda"`` or ``"cpu"``.
        **kwargs: Absorbed for API compatibility with ``get_cfg()`` dicts.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        resize: int,
        pad_color: int,
        ckpt_path: str,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.resize = resize
        self.pad_color = (pad_color, pad_color, pad_color)

        logger.info(
            "Loading model '%s' from '%s' on %s…", model_name, ckpt_path, self.device
        )

        try:
            raw = torch.load(ckpt_path, weights_only=True)
        except Exception:
            logger.warning(
                "Safe checkpoint loading failed for '%s'; retrying with weights_only=False. "
                "Only do this for checkpoints you trust.",
                ckpt_path,
            )
            raw = torch.load(ckpt_path, weights_only=False)

        if isinstance(raw, nn.Module):
            self.model = raw.to(self.device)
            self.model.eval()
            logger.info(
                "Loaded a full serialized model checkpoint directly from '%s'.",
                ckpt_path,
            )
            return

        lit = ImageClassificationModule(
            model_name=model_name, num_classes=num_classes
        )
        lit.custom_load_state_dict(ckpt_path)
        self.model = lit.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray, is_bgr: bool = True) -> torch.Tensor:
        """Convert a raw image array to a batched float tensor on device.

        Args:
            image: Numpy array ``(H, W, 3)``.
            is_bgr: ``True`` if the array is in BGR order (OpenCV default).

        Returns:
            Float tensor of shape ``(1, 3, resize, resize)`` on :attr:`device`.
        """
        if is_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = letterbox(image, self.resize, color=self.pad_color)
        tensor = torch.from_numpy(
            np.transpose(image / 255.0, (2, 0, 1))[None].astype(np.float32)
        )
        return tensor.to(self.device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image: np.ndarray, is_bgr: bool = True) -> tuple[int, float]:
        """Run inference on a single image.

        Args:
            image: Numpy array ``(H, W, 3)``.
            is_bgr: Whether the array is in BGR order.

        Returns:
            Tuple of ``(predicted_class_index, confidence_score)``.
        """
        tensor = self.preprocess(image, is_bgr=is_bgr)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)
        cls_idx: int = int(torch.argmax(probs, dim=1).item())
        score: float = float(probs[0, cls_idx].item())
        return cls_idx, score

    # ------------------------------------------------------------------
    # Batch runner (replaces old `forward()`)
    # ------------------------------------------------------------------

    def run(self, source: str) -> list[dict]:
        """Run inference over a file, directory, or glob pattern.

        Args:
            source: Path to a single image file or a directory of images.

        Returns:
            List of result dicts with keys ``"file"``, ``"class_index"``,
            ``"score"``.
        """
        paths = self._collect_paths(source)
        if not paths:
            logger.warning("No supported images found under '%s'.", source)
            return []

        results: list[dict] = []
        for fpath in paths:
            image = cv2.imread(fpath)
            if image is None:
                logger.warning("Could not read image '%s' — skipping.", fpath)
                continue
            cls_idx, score = self.predict(image, is_bgr=True)
            results.append({"file": fpath, "class_index": cls_idx, "score": score})
            logger.info("  %s  →  class=%d  score=%.4f", Path(fpath).name, cls_idx, score)

        return results

    @staticmethod
    def _collect_paths(source: str) -> list[str]:
        """Return sorted image paths from *source* (file or directory)."""
        if os.path.isfile(source):
            return [source] if Path(source).suffix.lower() in _SUPPORTED_EXTENSIONS else []
        if os.path.isdir(source):
            return sorted(
                str(p)
                for p in Path(source).rglob("*")
                if p.suffix.lower() in _SUPPORTED_EXTENSIONS
            )
        return []
