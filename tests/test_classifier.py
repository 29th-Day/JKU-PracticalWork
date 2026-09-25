from pathlib import Path
from typing import override

import lightning
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from project.classifier import ExoMinerClassifier
from project.features import Batch, UnlabeledBatch
from project.model import DIAGRAM, PAPER, REPO_V1_OPTIMIZER
from tests.helpers import random_batch


class _RandomBatches(Dataset[Batch]):
    """Pre-collated random batches, served one per index."""

    def __init__(self, batches: int, batch_size: int) -> None:
        self.batches: list[Batch] = [
            random_batch(PAPER, batch_size) for _ in range(batches)
        ]

    def __len__(self) -> int:
        return len(self.batches)

    @override
    def __getitem__(self, index: int) -> Batch:
        return self.batches[index]


def _loader(batches: int = 2, batch_size: int = 4) -> DataLoader[Batch]:
    return DataLoader(_RandomBatches(batches, batch_size), batch_size=None)


def _trainer(
        fast_dev_run: bool = False,
        max_steps: int = -1,
        default_root_dir: Path | None = None,
) -> lightning.Trainer:
    return lightning.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        fast_dev_run=fast_dev_run,
        max_steps=max_steps,
        default_root_dir=default_root_dir,
    )


def test_fast_dev_run_fits_validates_and_predicts() -> None:
    classifier = ExoMinerClassifier()
    trainer = _trainer(fast_dev_run=True)

    trainer.fit(classifier, _loader(), _loader())
    logged = set(trainer.callback_metrics)
    trainer.validate(classifier, _loader())
    predictions = trainer.predict(classifier, _loader())

    assert {"train/loss", "train/auc_pr", "val/loss", "val/auc_pr"} <= logged
    assert predictions is not None
    for probabilities in predictions:
        assert isinstance(probabilities, torch.Tensor)
        assert probabilities.shape == (4,)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_predict_without_labels() -> None:
    classifier = ExoMinerClassifier().eval()
    batch: UnlabeledBatch = {"features": random_batch(PAPER, 3)["features"]}

    probabilities = classifier.predict_step(batch, 0)

    assert probabilities.shape == (3,)


def test_validation_with_a_single_class() -> None:
    classifier = ExoMinerClassifier()
    dataset = _RandomBatches(batches=1, batch_size=4)
    for batch in dataset.batches:
        batch["label"] = torch.zeros(4)
    trainer = _trainer()

    with pytest.warns(UserWarning):
        results = trainer.validate(classifier, DataLoader(dataset, batch_size=None))

    assert "val/loss" in results[0]


def test_overfits_a_single_batch() -> None:
    classifier = ExoMinerClassifier()
    batch = random_batch(PAPER, 8)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    losses: list[float] = []

    for _ in range(30):
        optimizer.zero_grad()
        loss = loss_fn(classifier(batch["features"]), batch["label"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < 0.5 * losses[0]


def test_checkpoint_restores_config(tmp_path: Path) -> None:
    classifier = ExoMinerClassifier(DIAGRAM, REPO_V1_OPTIMIZER)
    trainer = _trainer(max_steps=1, default_root_dir=tmp_path)
    trainer.fit(classifier, _loader(batches=1), _loader(batches=1))
    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)

    restored = ExoMinerClassifier.load_from_checkpoint(path)

    assert restored.model.config == DIAGRAM
    assert restored.optimizer_config == REPO_V1_OPTIMIZER
