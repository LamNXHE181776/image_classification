"""AICycle Image Classification — top-level Python API.

The :class:`Classifier` class is the single entry-point for all tasks,
modelled after the Ultralytics ``YOLO`` API.

Quick-start::

    from classify import Classifier

    # ── Training ──────────────────────────────────────────────────────
    clf = Classifier("cfg/car_quality.yaml")
    clf.train()

    # ── Override any config key on the fly ────────────────────────────
    clf.train(data_dir="/new/dataset", max_epochs=100)

    # ── Prediction ────────────────────────────────────────────────────
    results = clf.predict(source="/path/to/images", ckpt_path="best.ckpt")
    for r in results:
        print(r["file"], r["class_index"], r["score"])

    # ── ONNX Export ───────────────────────────────────────────────────
    clf.export(ckpt_path="best.ckpt", file_path="model.onnx")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from classify.engine.exporter import export_onnx, export_torchscript
from classify.engine.predictor import Predictor
from classify.engine.trainer import ClassificationTrainer
from classify.models.registry import register_model, list_models
from classify.utils.config import get_cfg
from classify.utils.logging import get_logger

logger = get_logger(__name__)

__version__ = "1.0.0"
__all__ = ["Classifier", "register_model", "list_models", "__version__"]


class Classifier:
    """High-level façade for train / predict / export workflows.

    Args:
        cfg: One of:

            * ``None``              — use ``cfg/default.yaml``.
            * ``str`` / ``Path``    — path to a task-specific YAML.
            * ``dict``              — config overrides applied on top of defaults.

    Example::

        # Minimal: all settings come from a YAML file
        clf = Classifier("cfg/car_shape.yaml")
        clf.train()

        # Fluent overrides without touching the YAML
        clf.train(max_epochs=200, batch_size=32)
    """

    def __init__(self, cfg: dict[str, Any] | str | Path | None = None) -> None:
        self.cfg = get_cfg(cfg)
        logger.info(
            "Classifier ready — model=%s  classes=%d",
            self.cfg.get("model_name"),
            self.cfg.get("num_classes"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, **overrides: Any) -> None:
        """Train the model using the current configuration.

        Keyword arguments override individual config values for this run
        without mutating the stored ``self.cfg``.

        Args:
            **overrides: Any key from ``cfg/default.yaml``
                (e.g. ``data_dir=…``, ``max_epochs=100``).
        """
        cfg = {**self.cfg, **{k: v for k, v in overrides.items() if v is not None}}
        trainer = ClassificationTrainer(cfg)
        trainer.train()

    def predict(
        self,
        source: str,
        ckpt_path: str | None = None,
        device: str = "cuda",
        **overrides: Any,
    ) -> list[dict]:
        """Run inference on a file or directory of images.

        Args:
            source: Path to a single image or a directory of images.
            ckpt_path: Checkpoint to load.  Falls back to ``cfg["ckpt_path"]``.
            device: ``"cuda"`` or ``"cpu"``.
            **overrides: Additional config overrides (e.g. ``resize``, ``pad_color``).

        Returns:
            List of dicts with keys ``"file"``, ``"class_index"``, ``"score"``.

        Raises:
            ValueError: If no checkpoint path is provided or configured.
        """
        cfg = {**self.cfg, **{k: v for k, v in overrides.items() if v is not None}}
        resolved_ckpt = ckpt_path or cfg.get("ckpt_path", "")
        if not resolved_ckpt:
            raise ValueError(
                "A checkpoint path is required for prediction. "
                "Pass ckpt_path= or set ckpt_path in the config."
            )

        predictor = Predictor(
            model_name=cfg["model_name"],
            num_classes=cfg["num_classes"],
            resize=cfg["resize"],
            pad_color=cfg["pad_color"],
            ckpt_path=resolved_ckpt,
            device=device,
        )
        return predictor.run(source)

    def export(
        self,
        ckpt_path: str | None = None,
        file_path: str | None = None,
        format: str = "onnx",
        opset: int = 12,
        optimize_torchscript: bool = True,
        **overrides: Any,
    ) -> None:
        """Export a checkpoint to a portable deployment format.

        Supported formats:

        * ``"onnx"``        — cross-platform (ONNX Runtime, TensorRT, OpenCV DNN).
        * ``"torchscript"`` — PyTorch-native, no Python dependency at runtime.

        Args:
            ckpt_path: Source ``.ckpt`` file.  Falls back to ``cfg["ckpt_path"]``.
            file_path: Destination file path.  Defaults to ``model.onnx`` or
                ``model.pt`` depending on *format*.
            format: Export format — ``"onnx"`` (default) or ``"torchscript"``.
            opset: ONNX opset version (ignored for TorchScript).
            optimize_torchscript: Apply ``torch.jit.optimize_for_inference``
                after tracing (TorchScript only).
            **overrides: Additional config overrides (e.g. ``resize``).

        Raises:
            ValueError: If no checkpoint path is provided or configured, or if
                an unsupported *format* is given.
        """
        cfg = {**self.cfg, **{k: v for k, v in overrides.items() if v is not None}}
        resolved_ckpt = ckpt_path or cfg.get("ckpt_path", "")
        if not resolved_ckpt:
            raise ValueError(
                "A checkpoint path is required for export. "
                "Pass ckpt_path= or set ckpt_path in the config."
            )

        fmt = format.lower()
        if fmt == "onnx":
            dest = file_path or "model.onnx"
            export_onnx(
                ckpt_path=resolved_ckpt,
                file_path=dest,
                resize=cfg["resize"],
                opset=opset,
            )
        elif fmt in ("torchscript", "pt"):
            dest = file_path or "model.pt"
            export_torchscript(
                ckpt_path=resolved_ckpt,
                file_path=dest,
                resize=cfg["resize"],
                optimize=optimize_torchscript,
            )
        else:
            raise ValueError(
                f"Unsupported export format '{format}'. "
                "Choose 'onnx' or 'torchscript'."
            )

class SegmentationClassifier(Classifier):
    """High-level façade for image segmentation tasks.

    Inherits from :class:`Classifier` and adds segmentation-specific methods.
    """

    def train(self, **overrides: Any) -> None:
        """Train the segmentation model using the current configuration.

        Keyword arguments override individual config values for this run
        without mutating the stored ``self.cfg``.

        Args:
            **overrides: Any key from ``cfg/default.yaml``
                (e.g. ``data_dir=…``, ``max_epochs=100``).
        """
        cfg = {**self.cfg, **{k: v for k, v in overrides.items() if v is not None}}
        trainer = ClassificationTrainer(cfg, task_type="segmentation")
        trainer.train()