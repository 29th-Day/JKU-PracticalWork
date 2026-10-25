"""Lightning wrapper that trains ExoMiner++ as a binary planet classifier."""

__all__ = ["ExoMinerClassifier"]

from collections.abc import Mapping
from dataclasses import asdict
from typing import override

import lightning
import torch
from torch import Tensor
from torch.nn import functional
from torchmetrics import MetricCollection

from project.features import Batch, UnlabeledBatch
from project.metrics import paper_metrics
from project.model import (
    PAPER,
    PAPER_OPTIMIZER,
    ConfigValue,
    ExoMinerConfig,
    ExoMinerPlusPlus,
    OptimizerConfig,
    from_flat,
    to_flat,
)


class ExoMinerClassifier(lightning.LightningModule):
    """ExoMiner++ trained with binary cross-entropy and Adam (paper Sec. 5).

    Logs `{stage}/loss` and the paper metrics as `{stage}/<metric>` for the stages
    train, val and test.

    The configs are saved as hyperparameters in flat form (`to_flat`, tuples as
    lists), so checkpoints and `hparams.yaml` hold plain values keyed by name.
    `config` and `optimizer` accept that flat form too, which is how
    `load_from_checkpoint` rebuilds the module.
    """

    def __init__(
        self,
        config: ExoMinerConfig | Mapping[str, ConfigValue] = PAPER,
        optimizer: OptimizerConfig | Mapping[str, float] = PAPER_OPTIMIZER,
    ) -> None:
        super().__init__()
        if not isinstance(config, ExoMinerConfig):
            config = from_flat(config, PAPER)
        if not isinstance(optimizer, OptimizerConfig):
            optimizer = OptimizerConfig(**optimizer)
        flat = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in to_flat(config).items()
        }
        self.save_hyperparameters({"config": flat, "optimizer": asdict(optimizer)})
        self.optimizer_config: OptimizerConfig = optimizer
        self.model: ExoMinerPlusPlus = ExoMinerPlusPlus(config)
        metrics = paper_metrics()
        self.train_metrics: MetricCollection = metrics.clone(prefix="train/")
        self.val_metrics: MetricCollection = metrics.clone(prefix="val/")
        self.test_metrics: MetricCollection = metrics.clone(prefix="test/")

    @override
    def forward(self, features: Mapping[str, Tensor]) -> Tensor:
        return self.model(features)

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        return self._shared_step(batch, self.train_metrics, "train")

    @override
    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:
        return self._shared_step(batch, self.val_metrics, "val")

    @override
    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:
        return self._shared_step(batch, self.test_metrics, "test")

    @override
    def predict_step(self, batch: UnlabeledBatch, batch_idx: int) -> Tensor:
        return torch.sigmoid(self(batch["features"]))

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        config = self.optimizer_config
        return torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

    def _shared_step(
        self,
        batch: Batch,
        metrics: MetricCollection,
        stage: str,
    ) -> Tensor:
        logits = self(batch["features"])
        labels = batch["label"].float()
        loss = functional.binary_cross_entropy_with_logits(logits, labels)
        metrics.update(torch.sigmoid(logits), labels.long())
        self.log(
            f"{stage}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(labels),
        )
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss
