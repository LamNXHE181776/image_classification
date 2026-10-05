"""PyTorch Lightning module for image classification.

Wraps any backbone from :mod:`classify.models.registry` and provides
training / validation steps, metric logging, and optimiser setup.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchmetrics
import lightning as L
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR

from classify.models.registry import build_model
from classify.utils.logging import get_logger
from ultralytics.optim.muon import MuSGD

from lightning.pytorch.callbacks import BaseFinetuning

logger = get_logger(__name__)


class ImageClassificationModule(L.LightningModule):
    """LightningModule for image classification with any registered backbone.

    Args:
        model_name: Torchvision or custom registry model key.
        num_classes: Number of output classes.
        optimizer: ``torch.optim`` class name (e.g. ``"SGD"``, ``"Adam"``, ``"MuSGD"``).
        learning_rate: Initial learning rate.
        learning_rate_fraction: Multiplier so that
            ``lr_final = learning_rate * learning_rate_fraction``.
        loss_fn: ``torch.nn`` loss class name (e.g. ``"CrossEntropyLoss"``).
        freeze_backbone: If ``True``, set ``requires_grad=False`` for all backbone weights.
        **kwargs: Extra kwargs stored as hyperparameters (ignored at runtime).

    Example::

        module = ImageClassificationModule(
            model_name="efficientnet_v2_s",
            num_classes=4,
            optimizer="SGD",
            learning_rate=0.005,
        )
    """

    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        num_classes: int = 2,
        optimizer: str = "SGD",
        learning_rate: float = 0.005,
        learning_rate_fraction: float = 0.1,
        loss_fn: str = "CrossEntropyLoss",
        freeze_backbone: bool = False,
        class_weights: torch.Tensor | None = None,
        scheduler: str = "LambdaLR",
        hmn_ratio: float| None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.learning_rate_fraction = learning_rate_fraction
        self.freeze_backbone = freeze_backbone

        self.model: nn.Module = build_model(model_name, num_classes=num_classes)
        self.criterion: nn.Module = torch.nn.__dict__[loss_fn](weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.scheduler = scheduler
        self.hmn_ratio = hmn_ratio
        self.save_hyperparameters()

        #layer freezing for transfer learning:
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        #unfreeze the final classification layer:
        ''' 
        The exact name of the final layer varies across architectures. For now this is not integrated 
        into the code so it freezes all weights. 
            
        '''
        '''    
        if hasattr(self.model, "classifier"):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "fc"):
            for param in self.model.fc.parameters():
                param.requires_grad = True
        '''

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def custom_load_state_dict(self, ckpt_path: str) -> None:
        """Load weights with tolerance for mismatched layer shapes.

        Layers whose shapes differ between the checkpoint and the current
        model are skipped (current weights are retained) with a warning.

        Args:
            ckpt_path: Path to a ``.ckpt`` or raw state-dict ``.pt`` file.
        """
        current = self.state_dict()
        try:
            raw = torch.load(ckpt_path, weights_only=True)
        except Exception:
            logger.warning(
                "Safe checkpoint loading failed for '%s'; retrying with weights_only=False. "
                "Only do this for checkpoints you trust.",
                ckpt_path,
            )
            raw = torch.load(ckpt_path, weights_only=False)

        if isinstance(raw, dict):
            loaded = raw.get("state_dict", raw)
        elif isinstance(raw, torch.nn.Module):
            loaded = raw.state_dict()
        else:
            raise TypeError(
                f"Unsupported checkpoint type '{type(raw).__name__}' loaded from '{ckpt_path}'."
            )

        new_state: dict[str, torch.Tensor] = {}
        missing_keys: list[str] = []
        mismatched_keys: list[str] = []

        for key, cur_val in current.items():
            loaded_val = loaded.get(key)
            if loaded_val is None:
                missing_keys.append(key)
                new_state[key] = cur_val
                continue

            if loaded_val.size() != cur_val.size():
                mismatched_keys.append(key)
                logger.warning(
                    "Shape mismatch for '%s': expected %s, got %s — keeping current weights.",
                    key,
                    cur_val.size(),
                    loaded_val.size(),
                )
                new_state[key] = cur_val
                continue

            new_state[key] = loaded_val

        if missing_keys:
            logger.warning(
                "Checkpoint '%s' is missing %d parameter(s); keeping current values for those keys.",
                ckpt_path,
                len(missing_keys),
            )

        self.load_state_dict(new_state, strict=False)
        logger.info("Loaded pretrained weights from '%s'.", ckpt_path)

    # ------------------------------------------------------------------
    # Forward & shared step
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the backbone.

        Args:
            x: Float tensor of shape ``(B, 3, H, W)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        return self.model(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        """Compute loss + accuracy and log them for *stage*.

        Args:
            batch: Tuple of ``(images, labels)``.
            stage: ``"train"`` or ``"val"``.

        Returns:
            Scalar loss tensor.
        """
        x, y = batch
        logits = self.forward(x)

        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_acc": acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "val")
        current_lr: float = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "learning_rate",
            current_lr,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Configure the optimiser and linear-decay LR scheduler.

        Returns:
            Tuple of ``([optimizer], [scheduler_dict])`` for Lightning.
        """
        if self.optimizer_name.lower() == "musgd": #Add support for MuSGD optimizer from ultralytics, which has built-in momentum and weight decay handling.
            mu_parameters = []
            sgd_parameters = []
            
            for param in self.parameters():
                if not param.requires_grad:
                    logger.debug("Skipping frozen parameter '%s' from optimizer.", param.shape)
                    continue
                if param.ndim > 1:
                    mu_parameters.append(param)
                else:
                    sgd_parameters.append(param)
            
            logger.info(
                "Using MuSGD optimizer with %d muon parameters and %d SGD parameters.",
                len(mu_parameters),
                len(sgd_parameters),
            )
            
            param_groups = [
                {"params": mu_parameters, "use_muon": True, "momentum": 0.937, "weight_decay": 0.0005},
                {"params": sgd_parameters, "use_muon": False, "momentum": 0.937, "weight_decay": 0.0005},
            ]

            optimizer = MuSGD(param_groups, lr=self.learning_rate) #Fix a bug where muon was not being applied due to missing "use_muon" key. Whoops.
        else:
            optimizer = torch.optim.__dict__[self.optimizer_name](
                self.parameters(), lr=self.learning_rate
            )
        max_epochs: int = self.trainer.max_epochs or 1
        fraction = self.learning_rate_fraction

        if self.scheduler.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=self.learning_rate * fraction
            ) #Add support for cosine annealing scheduler, which can sometimes yield better results than linear decay.

        elif self.scheduler.lower() == "cosinerestart":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max_epochs // 3, T_mult=1, eta_min=self.learning_rate * fraction
            ) #Add support for cosine annealing with restarts, which can help escape local minima and improve convergence.
            
        else:
            if self.scheduler.lower() not in ["LambdaLR", "cosine", "cosinerestarts"]:
                logger.warning(
                    "Unsupported scheduler '%s'; defaulting to linear decay. Supported schedulers are: 'LambdaLR'"
                    ", 'Cosine' and 'CosineRestart'.",
                    self.scheduler,
                )
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: 1 - (epoch / max_epochs) * (1 - fraction),
            )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
