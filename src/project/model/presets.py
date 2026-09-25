"""Preset configs for the three reference descriptions of ExoMiner++.

- `PAPER`: the paper text (App. A, Table 18, Sec. 3.1); gaps are filled from the
  Zenodo diagram, then from NASA's v1.0.0 code.
- `DIAGRAM`: the Zenodo architecture diagram of the model trained for the paper.
- `REPO_V1`: NASA ExoMiner v1.0.0 (`models_keras.py`, `exominer_plus_plus.yaml`).
"""

from dataclasses import replace

from project.model.config import (
    ConvBranchConfig,
    ExoMinerConfig,
    OptimizerConfig,
    ScalarBranchConfig,
    TransitViewsConfig,
    UnfoldedFluxConfig,
)

__all__ = [
    "DIAGRAM",
    "PAPER",
    "PAPER_OPTIMIZER",
    "REPO_V1",
    "REPO_V1_OPTIMIZER",
]

_STELLAR_SCALARS = (
    "tce_sdens_norm",
    "tce_steff_norm",
    "tce_smet_norm",
    "tce_slogg_norm",
    "tce_smass_norm",
    "tce_sradius_norm",
)
_DV_TCE_FIT_SCALARS = (
    "boot_fap_norm",
    "tce_cap_stat_norm",
    "tce_hap_stat_norm",
    "tce_period_norm",
    "tce_max_mult_ev_norm",
    "tce_max_sngle_ev_norm",
    "tce_robstat_norm",
    "tce_model_chisq_norm",
    "tce_prad_norm",
)
_DIKCO_SCALARS = ("tce_dikco_msky_norm", "tce_dikco_msky_err_norm")

PAPER = ExoMinerConfig(
    activation="relu",
    conv_init="he",
    branch_units=4,
    branch_dropout=0.1211,
    scalar_branch_dropout=True,
    head_layers=4,
    head_units=512,
    head_dropout=0.0215,
    global_flux=ConvBranchConfig(
        enabled=True,
        scalars=("flux_global_stat_abs_min_norm",),
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=5,
        pool_size=8,
    ),
    flux_trend=ConvBranchConfig(
        enabled=True,
        scalars=("flux_trend_global_stat_max_norm", "flux_trend_global_stat_min_norm"),
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=5,
        pool_size=8,
    ),
    flux_periodogram=ConvBranchConfig(
        enabled=True,
        scalars=("pgram_smooth_max_power_norm", "pgram_tpm_smooth_max_power_norm"),
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=5,
        pool_size=8,
    ),
    local_centroid=ConvBranchConfig(
        enabled=True,
        scalars=(*_DIKCO_SCALARS, "mag_shift_norm", "ruwe_norm"),
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=6,
        pool_size=4,
    ),
    momentum_dump=ConvBranchConfig(
        enabled=True,
        scalars=(),
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=6,
        pool_size=4,
    ),
    transit_views=TransitViewsConfig(
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=6,
        pool_size=4,
        flux_enabled=True,
        flux_scalars=("flux_local_stat_abs_min",),
        weak_secondary_enabled=True,
        weak_secondary_scalars=(
            "tce_albedo_stat_norm",
            "tce_ptemp_stat_norm",
            "tce_maxmes_norm",
            "flux_weak_secondary_local_stat_abs_min_norm",
        ),
        odd_even_enabled=True,
        odd_even_scalars=(
            "flux_odd_local_stat_abs_min_norm",
            "flux_even_local_stat_abs_min_norm",
        ),
    ),
    unfolded_flux=UnfoldedFluxConfig(
        enabled=True,
        scalars=("tce_num_transits_obs_norm", "tce_num_transits_norm"),
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=6,
        transit_kernel_size=1,
        pool_size=4,
        stat_filters=4,
        stat_activation=True,
    ),
    diff_img=ConvBranchConfig(
        enabled=True,
        scalars=("mag_shift_norm",),
        blocks=3,
        layers_per_block=3,
        init_filters=4,
        kernel_size=3,
        pool_size=2,
    ),
    stellar=ScalarBranchConfig(enabled=True, scalars=_STELLAR_SCALARS),
    dv_tce_fit=ScalarBranchConfig(enabled=True, scalars=_DV_TCE_FIT_SCALARS),
)

DIAGRAM = replace(
    PAPER,
    activation="prelu",
    branch_units=3,
    scalar_branch_dropout=False,
    flux_trend=replace(PAPER.flux_trend, pool_size=4),
    flux_periodogram=replace(PAPER.flux_periodogram, pool_size=4),
    unfolded_flux=replace(PAPER.unfolded_flux, stat_activation=False),
)

REPO_V1 = replace(
    PAPER,
    activation="prelu",
    conv_init="glorot",
    branch_units=3,
    branch_dropout=0.1211390996398814,
    scalar_branch_dropout=False,
    head_dropout=0.021491238286347598,
    global_flux=replace(PAPER.global_flux, layers_per_block=5),
    flux_trend=replace(PAPER.flux_trend, pool_size=4),
    flux_periodogram=replace(PAPER.flux_periodogram, pool_size=4),
    local_centroid=replace(
        PAPER.local_centroid,
        scalars=("mag_shift_norm",),
        kernel_size=5,
    ),
    momentum_dump=replace(PAPER.momentum_dump, kernel_size=5),
    transit_views=replace(PAPER.transit_views, layers_per_block=1, init_filters=4),
    unfolded_flux=replace(
        PAPER.unfolded_flux,
        kernel_size=1,
        transit_kernel_size=6,
        stat_activation=False,
    ),
    diff_img=replace(PAPER.diff_img, scalars=("mag_shift_norm", *_DIKCO_SCALARS)),
    stellar=replace(PAPER.stellar, scalars=(*_STELLAR_SCALARS, "ruwe_norm")),
)

PAPER_OPTIMIZER = OptimizerConfig(
    learning_rate=4.18e-5,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
)

REPO_V1_OPTIMIZER = replace(PAPER_OPTIMIZER, learning_rate=4.176171931455995e-05)
