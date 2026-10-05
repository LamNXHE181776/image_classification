"""Lightning callbacks used during classification training."""

from __future__ import annotations

import os
from glob import glob

import mlflow
from lightning.pytorch.callbacks import Callback

from classify.engine.exporter import export_onnx
from classify.utils.logging import get_logger

logger = get_logger(__name__)


class GenerateLabelFileCallback(Callback):
    """Write a ``labels.txt`` file at the start of training.

    The file lists all class names in alphabetical order (one per line),
    matching the integer indices used at inference time.

    Args:
        data_dir: Dataset root containing a ``train/`` sub-folder.
        save_output_dir: Directory where ``labels.txt`` will be written.
    """

    def __init__(self, data_dir: str, save_output_dir: str) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.save_output_dir = save_output_dir

    def on_train_start(self, trainer, pl_module) -> None:
        train_dir = os.path.join(self.data_dir, "train")
        labels = sorted(
            os.path.basename(p)
            for p in glob(os.path.join(train_dir, "*"))
            if os.path.isdir(p)
        )
        os.makedirs(self.save_output_dir, exist_ok=True)
        label_file = os.path.join(self.save_output_dir, "labels.txt")
        with open(label_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(labels) + "\n")
        logger.info("Labels file written to '%s'.", label_file)


def _list_files(directory: str) -> list[str]:
    """Return absolute paths of all *files* directly inside *directory*."""
    return [
        os.path.abspath(p)
        for p in glob(os.path.join(directory, "*"))
        if os.path.isfile(p)
    ]


def _latest_version_dir(save_output_dir: str) -> str | None:
    """Return the path of the highest-numbered ``version_N`` sub-folder."""
    versions = sorted(
        d for d in os.listdir(save_output_dir) if d.startswith("version_")
    )
    return os.path.join(save_output_dir, versions[-1]) if versions else None


class LogArtifactsToMLflowCallback(Callback):
    """Upload checkpoints, CSVs, and labels to the active MLflow run.

    At the end of training the best ``.ckpt`` is also converted to ONNX and
    logged as an artifact.

    Args:
        save_output_dir: CSV-logger output directory for this run.
        resize: Image size used for the ONNX input shape.
    """

    def __init__(self, save_output_dir: str, resize: int) -> None:
        super().__init__()
        self.save_output_dir = save_output_dir
        self.resize = resize

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mlflow_client(self, trainer) -> tuple[mlflow.tracking.MlflowClient, str] | tuple[None, None]:
        """Extract the MlflowClient and run-id from the trainer's loggers."""
        for lg in getattr(trainer, "loggers", []):
            if lg.__class__.__name__ == "MLFlowLogger":
                return mlflow.tracking.MlflowClient(), lg._run_id
        return None, None

    def _log_files(self, client: mlflow.tracking.MlflowClient, run_id: str, *dirs: str) -> None:
        """Upload all files in each of *dirs* to the MLflow run."""
        for directory in dirs:
            if not os.path.isdir(directory):
                continue
            for fpath in _list_files(directory):
                client.log_artifact(run_id, fpath)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        client, run_id = self._get_mlflow_client(trainer)
        if client is None:
            return

        version_dir = _latest_version_dir(self.save_output_dir)
        if version_dir:
            self._log_files(
                client, run_id,
                version_dir,
                self.save_output_dir,
                os.path.join(version_dir, "checkpoints"),
            )

    def on_train_end(self, trainer, pl_module) -> None:
        client, run_id = self._get_mlflow_client(trainer)
        if client is None:
            return

        version_dir = _latest_version_dir(self.save_output_dir)
        ckpt_dir = os.path.join(version_dir, "checkpoints") if version_dir else None

        # Export ONNX from best checkpoint
        if ckpt_dir and os.path.isdir(ckpt_dir):
            for fpath in _list_files(ckpt_dir):
                if fpath.endswith(".ckpt"):
                    onnx_path = fpath.replace(".ckpt", ".onnx")
                    logger.info("Exporting ONNX from '%s'…", fpath)
                    export_onnx(ckpt_path=fpath, file_path=onnx_path, resize=self.resize)
                    break

        if version_dir:
            self._log_files(
                client, run_id,
                version_dir,
                self.save_output_dir,
                ckpt_dir,
            )

        mlflow.end_run()
        logger.info("MLflow run '%s' finalised.", run_id)
