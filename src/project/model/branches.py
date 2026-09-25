"""The ExoMiner++ branches; each maps the inputs to a `(B, out_features)` tensor.

All convolutions are 2D. Time series are fed as `(B, channels, 1, bins)` with kernels
of height 1, so rows never mix. That lets the transit views stack as rows and share
one convolution, which is exactly the paper's shared feature extractor.
"""

from collections.abc import Mapping
from typing import override

import torch
from torch import Tensor, nn

from project.features import (
    DIFF_IMG_VIEWS,
    EVEN_VIEWS,
    IMAGE_SIZE,
    LOCAL_FLUX_VIEWS,
    ODD_VIEWS,
    QUALITY,
    SECTORS,
    TRANSIT_BINS,
    UNFOLDED_FLUX_VIEW,
    WEAK_SECONDARY_VIEWS,
)
from project.model.config import (
    Activation,
    ConvBranchConfig,
    ScalarBranchConfig,
    TransitViewsConfig,
    UnfoldedFluxConfig,
    pooled_length,
)
from project.model.layers import (
    BranchHead,
    ConvStack,
    FullWidthLinear,
    make_activation,
    with_scalars,
)

__all__ = [
    "Branch",
    "DifferenceImageBranch",
    "ScalarBranch",
    "TimeSeriesBranch",
    "TransitViewBranches",
    "UnfoldedFluxBranch",
]


class ScalarBranch(nn.Module):
    """Stellar parameters or DV fit statistics through one FC layer (Sec. 3.1, 8)."""

    def __init__(
            self,
            config: ScalarBranchConfig,
            units: int,
            activation: Activation,
            dropout: float,
    ) -> None:
        super().__init__()
        self.scalars: tuple[str, ...] = config.scalars
        in_features = len(config.scalars)
        self.head: BranchHead = BranchHead(in_features, units, activation, dropout)
        self.out_features: int = units

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        return self.head(torch.stack([inputs[name] for name in self.scalars], dim=1))


class TimeSeriesBranch(nn.Module):
    """A phase-folded view plus its uncertainty channel through a conv stack.

    Used for global flux, flux trend, periodogram, centroid and momentum dump.
    """

    def __init__(
            self,
            config: ConvBranchConfig,
            views: tuple[str, ...],
            bins: int,
            units: int,
            activation: Activation,
            dropout: float,
    ) -> None:
        super().__init__()
        self.views: tuple[str, ...] = views
        self.scalars: tuple[str, ...] = config.scalars
        self.convs: ConvStack = ConvStack(
            in_channels=len(views),
            blocks=config.blocks,
            layers_per_block=config.layers_per_block,
            init_filters=config.init_filters,
            kernel_size=(1, config.kernel_size),
            pool_size=(1, config.pool_size),
            activation=activation,
        )
        length = pooled_length(bins, config.blocks, config.pool_size)
        in_features = self.convs.out_channels * length + len(config.scalars)
        self.head: BranchHead = BranchHead(in_features, units, activation, dropout)
        self.out_features: int = units

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        series = torch.stack([inputs[name] for name in self.views], dim=1)
        maps = self.convs(series.unsqueeze(2))
        return self.head(with_scalars(maps, inputs, self.scalars))


class TransitViewBranches(nn.Module):
    """Flux, weak secondary and odd/even transit views through one shared extractor.

    The views are stacked as rows and convolved together (Sec. 3.1 item 2), then each
    diagnostic gets its own head. Odd and even feature maps are subtracted first.
    """

    def __init__(
            self,
            config: TransitViewsConfig,
            units: int,
            activation: Activation,
            dropout: float,
    ) -> None:
        super().__init__()
        self.config: TransitViewsConfig = config
        self.rows: dict[str, tuple[str, ...]] = {}
        if config.flux_enabled:
            self.rows["flux"] = LOCAL_FLUX_VIEWS
        if config.weak_secondary_enabled:
            self.rows["weak_secondary"] = WEAK_SECONDARY_VIEWS
        if config.odd_even_enabled:
            self.rows["odd"] = ODD_VIEWS
            self.rows["even"] = EVEN_VIEWS
        self.convs: ConvStack = ConvStack(
            in_channels=2,
            blocks=config.blocks,
            layers_per_block=config.layers_per_block,
            init_filters=config.init_filters,
            kernel_size=(1, config.kernel_size),
            pool_size=(1, config.pool_size),
            activation=activation,
        )
        length = pooled_length(TRANSIT_BINS, config.blocks, config.pool_size)
        per_view = self.convs.out_channels * length
        self.heads: nn.ModuleDict = nn.ModuleDict()
        for name, enabled, scalars in self._diagnostics():
            if enabled:
                self.heads[name] = BranchHead(
                    per_view + len(scalars), units, activation, dropout
                )
        self.out_features: int = units * len(self.heads)

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        rows = [
            torch.stack([inputs[name] for name in views], dim=1)
            for views in self.rows.values()
        ]
        maps = self.convs(torch.stack(rows, dim=2))
        by_row = dict(zip(self.rows, maps.unbind(dim=2), strict=True))
        if self.config.odd_even_enabled:
            by_row["odd_even"] = by_row["odd"] - by_row["even"]
        outputs = [
            self.heads[name](with_scalars(by_row[name], inputs, scalars))
            for name, enabled, scalars in self._diagnostics()
            if enabled
        ]
        return torch.cat(outputs, dim=1)

    def _diagnostics(self) -> list[tuple[str, bool, tuple[str, ...]]]:
        config = self.config
        return [
            ("flux", config.flux_enabled, config.flux_scalars),
            (
                "weak_secondary",
                config.weak_secondary_enabled,
                config.weak_secondary_scalars,
            ),
            ("odd_even", config.odd_even_enabled, config.odd_even_scalars),
        ]


class UnfoldedFluxBranch(nn.Module):
    """The 20 individual transits, reduced to their min, max and mean (Sec. 4.2).

    A 1x1 convolution then mixes the three statistics per feature position.
    """

    def __init__(
            self,
            config: UnfoldedFluxConfig,
            units: int,
            activation: Activation,
            dropout: float,
    ) -> None:
        super().__init__()
        self.scalars: tuple[str, ...] = config.scalars
        self.convs: ConvStack = ConvStack(
            in_channels=1,
            blocks=config.blocks,
            layers_per_block=config.layers_per_block,
            init_filters=config.init_filters,
            kernel_size=(config.transit_kernel_size, config.kernel_size),
            pool_size=(1, config.pool_size),
            activation=activation,
        )
        self.stat_conv: nn.Conv2d = nn.Conv2d(3, config.stat_filters, kernel_size=1)
        self.stat_activation: nn.Module = (
            make_activation(activation) if config.stat_activation else nn.Identity()
        )
        length = pooled_length(TRANSIT_BINS, config.blocks, config.pool_size)
        stat_features = config.stat_filters * self.convs.out_channels * length
        in_features = stat_features + len(config.scalars)
        self.head: BranchHead = BranchHead(in_features, units, activation, dropout)
        self.out_features: int = units

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        maps = self.convs(inputs[UNFOLDED_FLUX_VIEW].unsqueeze(1))
        stats = torch.stack([maps.amin(dim=2), maps.amax(dim=2), maps.mean(dim=2)], 1)
        mixed = self.stat_activation(self.stat_conv(stats))
        return self.head(with_scalars(mixed, inputs, self.scalars))


class DifferenceImageBranch(nn.Module):
    """Difference, out-of-transit and target images of 5 sectors (Sec. 4.2).

    Each sector is convolved on its own with shared weights. The feature maps of all
    sectors and their quality metrics then go through one full-width layer, as in
    NASA's code, before the scalars are appended.
    """

    def __init__(
            self,
            config: ConvBranchConfig,
            units: int,
            activation: Activation,
            dropout: float,
    ) -> None:
        super().__init__()
        self.scalars: tuple[str, ...] = config.scalars
        self.convs: ConvStack = ConvStack(
            in_channels=len(DIFF_IMG_VIEWS),
            blocks=config.blocks,
            layers_per_block=config.layers_per_block,
            init_filters=config.init_filters,
            kernel_size=(config.kernel_size, config.kernel_size),
            pool_size=(config.pool_size, config.pool_size),
            activation=activation,
            pool_every_layer=True,
        )
        pools = config.blocks * config.layers_per_block
        size = pooled_length(IMAGE_SIZE, pools, config.pool_size)
        sector_features = SECTORS * self.convs.out_channels * size * size + SECTORS
        self.sectors_layer: FullWidthLinear = FullWidthLinear(sector_features, units)
        self.sectors_activation: nn.Module = make_activation(activation)
        in_features = units + len(config.scalars)
        self.head: BranchHead = BranchHead(in_features, units, activation, dropout)
        self.out_features: int = units

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        images = torch.stack([inputs[name] for name in DIFF_IMG_VIEWS], dim=2)
        maps = self.convs(images.flatten(0, 1))
        sectors = torch.cat([maps.reshape(len(images), -1), inputs[QUALITY]], dim=1)
        summary = self.sectors_activation(self.sectors_layer(sectors))
        return self.head(with_scalars(summary, inputs, self.scalars))


Branch = (
    ScalarBranch
    | TimeSeriesBranch
    | TransitViewBranches
    | UnfoldedFluxBranch
    | DifferenceImageBranch
)
