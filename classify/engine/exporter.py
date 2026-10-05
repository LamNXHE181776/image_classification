"""Export utilities for trained classification models.

Supported formats:
    - **ONNX** (``export_onnx``) — cross-platform deployment (OpenCV DNN,
      TensorRT, ONNX Runtime, Core ML via onnx-coreml, …).
    - **TorchScript** (``export_torchscript``) — portable PyTorch format,
      no Python dependency at runtime.
"""

from __future__ import annotations

from pathlib import Path

import torch

from classify.models.lightning_module import ImageClassificationModule
from classify.utils.logging import get_logger
from torch.export import export
logger = get_logger(__name__)


def export_onnx(
    ckpt_path: str,
    file_path: str,
    resize: int,
    input_name: str = "input",
    output_name: str = "output",
    opset: int = 12,
) -> None:
    """Export a trained classification checkpoint to ONNX format.

    Args:
        ckpt_path: Path to a Lightning ``.ckpt`` checkpoint.
        file_path: Destination path for the exported ``.onnx`` file.
        resize: Spatial dimension used as the ONNX input shape
            ``(1, 3, resize, resize)``.
        input_name: Name of the ONNX input node.
        output_name: Name of the ONNX output node.
        opset: ONNX opset version (default: 12).

    Raises:
        FileNotFoundError: If *ckpt_path* does not exist.
    """
    logger.info("Loading checkpoint from '%s'…", ckpt_path)
    module = ImageClassificationModule.load_from_checkpoint(ckpt_path)
    module.eval()

    dummy_input = torch.randn(1, 3, resize, resize, dtype=torch.float32)
    logger.info(
        "Exporting ONNX to '%s' (opset %d, input shape %s)…",
        file_path,
        opset,
        tuple(dummy_input.shape),
    )

    outpath = Path(file_path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    save_path = outpath.with_suffix(".onnx")
    module.to_onnx(
        file_path=str(save_path),
        input_sample=dummy_input,
        export_params=True,
        opset_version=opset,
        input_names=[input_name],
        output_names=[output_name],
    )
    logger.info("ONNX export complete → '%s'.", file_path)

def export_torchscript(
    ckpt_path: str,
    file_path: str,
    resize: int,
    optimize: bool = True,
    export_format: str = "trace",
    device: str = "cpu",
) -> None:
    """Export a trained classification checkpoint to TorchScript format.

    The resulting ``.pt`` file can be loaded in C++ or Python without the
    original class definition, making it ideal for server-side deployment.

    Args:
        ckpt_path: Path to a Lightning ``.ckpt`` checkpoint.
        file_path: Destination path for the exported ``.pt`` file.
        resize: Spatial dimension of the dummy input used for tracing
            ``(1, 3, resize, resize)``.
        optimize: Apply ``torch.jit.optimize_for_inference`` after tracing
            (default: ``True``).
        export_format: Type of the exported ``.pt`` file (default: ``script``). Options: pt2, trace.
        device: "cuda" or "cpu" (default: "cuda" if available).

    Raises:
        FileNotFoundError: If *ckpt_path* does not exist.
    """
    logger.info("Loading checkpoint from '%s'…", ckpt_path)

    module = ImageClassificationModule.load_from_checkpoint(ckpt_path)
    module.eval()
    dummy_input = torch.randn(1, 3, resize, resize, dtype=torch.float32).to(device)
    module.to(device)

    logger.info(
        "Tracing TorchScript to '%s' (input shape %s)…",
        file_path,
        tuple(dummy_input.shape),
    )

    outpath = Path(file_path)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if export_format == "pt2":
        traced = export(module, (dummy_input,), strict=False)
        save_path = outpath.with_suffix(".pt2")
        torch.export.save(traced, str(save_path))

    elif export_format == "trace":
        save_path = outpath.with_suffix(".pt")
        if optimize:
            with torch.jit.optimized_execution(True):
                module.to_torchscript(
                    file_path=str(save_path),
                    method="trace",
                    example_inputs=dummy_input,
                    optimize=optimize,
                )
        else:
            module.to_torchscript(
                file_path=str(save_path),
                method="trace",
                example_inputs=dummy_input,
                optimize=optimize,
            )

    logger.info("TorchScript export complete → '%s'.", file_path)



