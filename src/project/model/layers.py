"""Building blocks shared by the ExoMiner++ branches."""

__all__ = [
    "BranchHead",
    "ConvStack",
    "FullWidthLinear",
    "init_weights",
    "make_activation",
    "with_scalars",
]

from collections.abc import Mapping, Sequence
from typing import override

import torch
from torch import Tensor, nn

from project.model.config import Activation, ConvInit


def make_activation(activation: Activation) -> nn.Module:
    """ReLU, or PReLU with a single slope initialized to 0 as in NASA's Keras code."""
    if activation == "prelu":
        return nn.PReLU(init=0.0)
    return nn.ReLU()


class ConvStack(nn.Sequential):
    """Blocks of `layers_per_block` x (Conv2d -> activation), each followed by a pool.

    Block `i` has `init_filters * 2**i` filters. Convolutions use 'same' padding and
    stride 1; max-pools use 'valid' padding and stride 1. With `pool_every_layer`,
    every convolution is followed by a pool instead of every block.
    """

    def __init__(
            self,
            in_channels: int,
            blocks: int,
            layers_per_block: int,
            init_filters: int,
            kernel_size: tuple[int, int],
            pool_size: tuple[int, int],
            activation: Activation,
            pool_every_layer: bool = False,
    ) -> None:
        layers: list[nn.Module] = []
        channels = in_channels
        for block in range(blocks):
            filters = init_filters * 2**block
            for _ in range(layers_per_block):
                layers.append(nn.Conv2d(channels, filters, kernel_size, padding="same"))
                layers.append(make_activation(activation))
                if pool_every_layer:
                    layers.append(nn.MaxPool2d(pool_size, stride=1))
                channels = filters
            if not pool_every_layer:
                layers.append(nn.MaxPool2d(pool_size, stride=1))
        super().__init__(*layers)
        self.out_channels: int = channels


class BranchHead(nn.Module):
    """Linear -> activation -> dropout that ends every branch."""

    def __init__(
            self,
            in_features: int,
            units: int,
            activation: Activation,
            dropout: float,
    ) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(in_features, units)
        self.activation: nn.Module = make_activation(activation)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.linear(x)))


class FullWidthLinear(nn.Linear):
    """Linear layer standing in for NASA's full-width Conv1D; initialized as a conv."""


def with_scalars(
        features: Tensor,
        inputs: Mapping[str, Tensor],
        scalars: Sequence[str],
) -> Tensor:
    """Flattens `features` to `(B, n)` and appends the named scalars as columns."""
    columns = [features.flatten(1), *(inputs[name].unsqueeze(1) for name in scalars)]
    return torch.cat(columns, dim=1)


def init_weights(module: nn.Module, conv_init: ConvInit) -> None:
    """Initializes convs per `conv_init`, linear layers with Xavier, biases to 0."""
    if not isinstance(module, nn.Conv2d | nn.Linear):
        return
    is_conv = isinstance(module, nn.Conv2d | FullWidthLinear)
    if is_conv and conv_init == "he":
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
