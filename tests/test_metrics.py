import subprocess
import sys

import pytest
import torch

from project.metrics import paper_metrics


def test_paper_metrics_on_hand_computed_example() -> None:
    metrics = paper_metrics()
    probabilities = torch.tensor([0.9, 0.8, 0.3, 0.6, 0.1])
    targets = torch.tensor([1, 1, 1, 0, 0])

    result = metrics(probabilities, targets)

    assert result["accuracy"].item() == pytest.approx(3 / 5)
    assert result["precision"].item() == pytest.approx(2 / 3)
    assert result["recall"].item() == pytest.approx(2 / 3)
    assert result["auc_roc"].item() == pytest.approx(5 / 6)
    assert result["auc_pr"].item() == pytest.approx((1 + 1 + 3 / 4) / 3)


def test_shared_modules_do_not_import_the_model() -> None:
    code = (
        "import sys, project.features, project.metrics; "
        "assert 'project.model' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", code], check=True)
