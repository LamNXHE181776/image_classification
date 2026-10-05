"""Training orchestration for image classification.

Wires together the DataModule, LightningModule, callbacks, and loggers into
a single :class:`ClassificationTrainer` that mirrors the public API of the
top-level :class:`~classify.Classifier`.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

import lightning as L
import pytz
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger

from classify.data.dataset import ClassificationDataModule
from classify.data.utils import auto_num_classes, get_dataset_classes
from classify.engine.callbacks import (
    GenerateLabelFileCallback,
    LogArtifactsToMLflowCallback,
)
from classify.models.lightning_module import ImageClassificationModule
from classify.utils.logging import get_logger


logger = get_logger(__name__)

_VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")

class PyCharmProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self) -> Tqdm:
        bar = super().init_validation_tqdm()
        bar.disable = True  # Completely silences the validation sub-bar loop
        return bar

class ClassificationTrainer:
    """Orchestrates training of an image classification model.

    All configuration is consumed from a single dict (typically produced by
    :func:`~classify.utils.config.get_cfg`), so no individual keyword
    argument is hard-coded in Python.

    Args:
        cfg: Resolved configuration dict.  See ``cfg/default.yaml`` for the
            full list of supported keys.

    Example::

        from classify.utils.config import get_cfg
        from classify.engine.trainer import ClassificationTrainer

        cfg = get_cfg("cfg/car_quality.yaml")
        trainer = ClassificationTrainer(cfg)
        trainer.train()
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = dict(cfg)  # work on a local copy so we can mutate safely
        self._resolve_num_classes()
        today = date.today().strftime("%Y%m%d")
        self.run_dir = os.path.join(
            self.cfg["save_output_dir"],
            today,
            self.cfg["project"],
            self.cfg["model_name"],
        )

    def _resolve_num_classes(self) -> None:
        """Detect ``num_classes`` from the dataset and validate / set in cfg.

        Always reads the actual class folders from ``<data_dir>/train/`` so
        that any mismatch between the configured value and the real data is
        caught **before** training starts (not as a silent CUDA OOB crash).

        Behaviour:
            * ``num_classes == 0`` or not set → auto-fill from dataset, log ``[AUTO]``.
            * ``num_classes > 0`` and matches dataset → log ``[OK]``.
            * ``num_classes > 0`` but **mis-matches** dataset → raise ``ValueError``
              so the user sees a clear message instead of a CUDA error.

        Raises:
            ValueError: If ``data_dir`` is empty/missing, or if an explicit
                ``num_classes`` does not match the actual dataset class count.
        """
        data_dir = self.cfg.get("data_dir", "")
        if not data_dir:
            raise ValueError(
                "data_dir is empty. Provide a valid data_dir in the config."
            )

        classes = get_dataset_classes(data_dir)
        detected = len(classes)
        configured = self.cfg.get("num_classes") or 0

        if configured == 0:
            # Auto-fill
            self.cfg["num_classes"] = detected
            logger.info(
                "[AUTO] num_classes not set — detected %d class(es) from '%s/train/': %s",
                detected,
                data_dir,
                classes,
            )
        elif configured != detected:
            raise ValueError(
                f"num_classes mismatch: config says {configured} but "
                f"'{data_dir}/train/' has {detected} class folder(s): {classes}.\n"
                f"Fix: set num_classes={detected} (or num_classes=0 for auto-detect)."
            )
        else:
            logger.info(
                "[OK] num_classes=%d matches dataset (%d class folder(s) in '%s/train/'): %s",
                configured,
                detected,
                data_dir,
                classes,
            )

    # ------------------------------------------------------------------
    # Factory methods — each returns a well-typed Lightning object
    # ------------------------------------------------------------------

    def _build_data_module(self) -> ClassificationDataModule:
        cfg = self.cfg
        return ClassificationDataModule(
            data_dir=cfg["data_dir"],
            resize=cfg["resize"],
            batch_size=cfg["batch_size"],
            apply_aug=cfg["apply_aug"],
            num_workers=cfg["num_workers"],
            pad_color=cfg["pad_color"],
            use_class_weights=cfg["use_class_weights"],
        )

    def _build_model(self, class_weights: np.ndarray | None = None) -> ImageClassificationModule:
        cfg = self.cfg
        
        # Convert class weights to tensor if they exist
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.from_numpy(class_weights)
            
        return ImageClassificationModule(
            model_name=cfg["model_name"],
            num_classes=cfg["num_classes"],
            optimizer=cfg["optimizer"],
            learning_rate=cfg["learning_rate"],
            learning_rate_fraction=cfg["learning_rate_fraction"],
            loss_fn=cfg["loss_fn"],
            freeze_backbone=cfg["freeze_backbone"],
            class_weights=weight_tensor,
            scheduler=cfg["scheduler"],
        )

    def _build_callbacks(self) -> list:
        cfg = self.cfg
        return [
            GenerateLabelFileCallback(cfg["data_dir"], self.run_dir),
            PyCharmProgressBar(refresh_rate=1),
            ModelCheckpoint(
                monitor="val_loss",
                filename="best_loss_{epoch}-{val_loss:.3f}-{val_acc:.3f}",
                save_top_k=1,
                save_last=True,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_acc",
                filename="best_acc_{epoch}-{val_loss:.3f}-{val_acc:.3f}",
                save_top_k=1,
                save_last=False,
                mode="max",
            ),

            EarlyStopping(
                monitor="val_loss",
                min_delta=0.0,
                patience=cfg["patience"],
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            LogArtifactsToMLflowCallback(
                save_output_dir=self.run_dir,
                resize=cfg["resize"],
            ),
        ]

    def _build_loggers(self) -> list:
        cfg = self.cfg
        loggers = [
            CSVLogger(
                save_dir=cfg["save_output_dir"],
                name=os.path.join(
                    date.today().strftime("%Y%m%d"),
                    cfg["project"],
                    cfg["model_name"],
                ),
            )
        ]
        if cfg.get("enable_mlflow", False):
            now = datetime.now(_VN_TZ).strftime("%Y-%m-%d_%H:%M:%S")
            loggers.append(
                MLFlowLogger(
                    experiment_name=cfg["project"],
                    run_name=f"{cfg['model_name']}_{now}",
                    tracking_uri=cfg["mlflow_tracking_uri"],
                )
            )
            logger.info(
                "MLflow logging enabled → %s", cfg["mlflow_tracking_uri"]
            )
        return loggers

    def _build_trainer(self) -> L.Trainer:
        cfg = self.cfg
        return L.Trainer(
            accelerator=cfg["accelerator"],
            devices=cfg["devices"],
            strategy=cfg["strategy"],
            precision=cfg["precision"],
            min_epochs=cfg["min_epochs"],
            max_epochs=cfg["max_epochs"],
            enable_progress_bar=True,
            logger=self._build_loggers(),
            callbacks=self._build_callbacks(),

        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training + validation loop.

        If ``cfg["ckpt_path"]`` points to an existing file the weights are
        loaded before training starts (fine-tuning / resume).
        """
        torch.set_float32_matmul_precision("medium")

        data_module = self._build_data_module()

        # ───────────── Call setup() early to get class weights and dataset stats before building the model ─────────────
        data_module.setup()

        model = self._build_model(data_module.class_weights)
        trainer = self._build_trainer()

        ckpt = self.cfg.get("ckpt_path", "")
        if ckpt and os.path.isfile(ckpt):
            logger.info("Resuming from checkpoint '%s'.", ckpt)
            model.custom_load_state_dict(ckpt)

        # ── Dataset stats (after setup so counts are available) ──────────────
        train_total = len(data_module.train_ds)
        val_total = len(data_module.val_ds)
        classes = list(data_module.train_ds.label_to_index.keys())
        per_class_train = {
            lbl: len(data_module.train_ds.X) // max(1, len(classes))
            for lbl in classes
        }
        # Accurate per-class counts from the dataset dict used at build time
        from classify.data.utils import create_dataset_dict
        train_dict, label_to_index = create_dataset_dict(
            os.path.join(self.cfg["data_dir"], "train")
        )
        val_dict, _ = create_dataset_dict(
            os.path.join(self.cfg["data_dir"], "val")
        )
        logger.info(
            "Dataset summary  |  classes=%d  train=%d  val=%d",
            self.cfg["num_classes"],
            train_total,
            val_total,
        )
        for lbl in sorted(label_to_index.keys()):
            t_count = len(train_dict.get(lbl, []))
            v_count = len(val_dict.get(lbl, []))
            logger.info(
                "  [%2d] %-30s  train=%5d  val=%5d",
                label_to_index[lbl],
                lbl,
                t_count,
                v_count,
            )

        logger.info(
            "Starting training — model=%s  num_classes=%d  resize=%d  epochs=%d",
            self.cfg["model_name"],
            self.cfg["num_classes"],
            self.cfg["resize"],
            self.cfg["max_epochs"],
        )
        # Pass the already-setup data_module so Lightning doesn't call setup() again
        trainer.fit(model=model, datamodule=data_module)
        trainer.validate(model=model, datamodule=data_module)
        logger.info("Training complete.  Outputs saved to '%s'.", self.run_dir)
