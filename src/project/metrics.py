"""Classification metrics reported in the ExoMiner++ paper (Sec. 6.1)."""

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)

__all__ = ["paper_metrics"]


def paper_metrics(threshold: float = 0.5) -> MetricCollection:
    """Accuracy, precision and recall at `threshold`, plus PR AUC and ROC AUC.

    Expects predicted probabilities and integer targets (1 = planet).
    """
    return MetricCollection(
        {
            "accuracy": BinaryAccuracy(threshold=threshold),
            "precision": BinaryPrecision(threshold=threshold),
            "recall": BinaryRecall(threshold=threshold),
            "auc_pr": BinaryAveragePrecision(),
            "auc_roc": BinaryAUROC(),
        }
    )
