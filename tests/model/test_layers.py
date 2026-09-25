import pytest
import torch
from torch import nn

from project.model.config import pooled_length
from project.model.layers import (
    BranchHead,
    ConvStack,
    FullWidthLinear,
    init_weights,
    make_activation,
    with_scalars,
)


@pytest.mark.parametrize(
    ("length", "pools", "pool_size", "expected"),
    [(301, 2, 8, 287), (674, 2, 8, 660), (31, 2, 4, 25), (33, 9, 2, 24)],
)
def test_pooled_length(length: int, pools: int, pool_size: int, expected: int) -> None:
    assert pooled_length(length, pools, pool_size) == expected


def test_make_activation_prelu_starts_as_relu() -> None:
    activation = make_activation("prelu")
    x = torch.tensor([-2.0, 3.0])

    assert isinstance(activation, nn.PReLU)
    assert torch.equal(activation(x), torch.tensor([0.0, 3.0]))


def test_conv_stack_time_series_shape() -> None:
    stack = ConvStack(
        in_channels=2,
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=(1, 6),
        pool_size=(1, 4),
        activation="relu",
    )

    out = stack(torch.randn(3, 2, 4, 31))

    assert out.shape == (3, 16, 4, 25)
    assert stack.out_channels == 16
    assert sum(isinstance(layer, nn.Conv2d) for layer in stack) == 6


def test_conv_stack_pool_every_layer_shape() -> None:
    stack = ConvStack(
        in_channels=3,
        blocks=3,
        layers_per_block=3,
        init_filters=4,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        activation="relu",
        pool_every_layer=True,
    )

    assert stack(torch.randn(2, 3, 33, 33)).shape == (2, 16, 24, 24)


def test_conv_stack_even_kernel_keeps_rows_separate() -> None:
    stack = ConvStack(
        in_channels=1,
        blocks=1,
        layers_per_block=1,
        init_filters=1,
        kernel_size=(1, 6),
        pool_size=(1, 1),
        activation="relu",
    )
    x = torch.randn(1, 1, 2, 31)
    changed = x.clone()
    changed[:, :, 1] += 1.0

    out, out_changed = stack(x), stack(changed)

    assert out.shape == (1, 1, 2, 31)
    assert torch.equal(out[:, :, 0], out_changed[:, :, 0])


def test_branch_head_shape_and_dropout() -> None:
    head = BranchHead(in_features=10, units=4, activation="relu", dropout=0.5)

    assert head(torch.randn(3, 10)).shape == (3, 4)
    assert head.dropout.p == 0.5


def test_with_scalars_appends_columns() -> None:
    features = torch.zeros(2, 3, 5)
    inputs = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}

    combined = with_scalars(features, inputs, ("a", "b"))

    assert combined.shape == (2, 17)
    assert torch.equal(combined[:, -2:], torch.tensor([[1.0, 3.0], [2.0, 4.0]]))


def test_init_weights_zeroes_biases_and_uses_he_for_convs() -> None:
    conv = nn.Conv2d(64, 64, kernel_size=3)
    full_width = FullWidthLinear(4096, 64)
    linear = nn.Linear(4096, 64)

    for module in (conv, full_width, linear):
        init_weights(module, conv_init="he")

    assert conv.bias is not None
    assert torch.count_nonzero(conv.bias) == 0
    assert conv.weight.std().item() == pytest.approx((2 / (64 * 9)) ** 0.5, rel=0.1)
    assert full_width.weight.std().item() == pytest.approx((2 / 4096) ** 0.5, rel=0.1)
    xavier_std = (2 / (4096 + 64)) ** 0.5
    assert linear.weight.std().item() == pytest.approx(xavier_std, rel=0.1)
