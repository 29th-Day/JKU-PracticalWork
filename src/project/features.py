"""Feature names, shapes and the batch layout shared by the data code and the model.

Names follow NASA's ExoMiner++ TFRecord features. Shapes are per example; batching
adds a leading dimension. Scalars have shape `()`.
"""

__all__ = [
    "CENTROID_VIEWS",
    "DIFF_IMG_VIEWS",
    "EVEN_VIEWS",
    "FEATURE_SHAPES",
    "FLUX_TREND_VIEWS",
    "FULL_ORBIT_BINS",
    "GLOBAL_FLUX_VIEWS",
    "IMAGE_SIZE",
    "LOCAL_FLUX_VIEWS",
    "MOMENTUM_DUMP_VIEWS",
    "ODD_VIEWS",
    "PERIODOGRAM_BINS",
    "PERIODOGRAM_VIEWS",
    "QUALITY",
    "SCALAR_FEATURES",
    "SECTORS",
    "TRANSIT_BINS",
    "UNFOLDED_FLUX_VIEW",
    "UNFOLDED_TRANSITS",
    "WEAK_SECONDARY_VIEWS",
    "Batch",
    "UnlabeledBatch",
]

from typing import TypedDict

from torch import Tensor

TRANSIT_BINS = 31
FULL_ORBIT_BINS = 301
PERIODOGRAM_BINS = 674
UNFOLDED_TRANSITS = 20
SECTORS = 5
IMAGE_SIZE = 33

GLOBAL_FLUX_VIEWS = ("global_flux_view_fluxnorm", "global_flux_view_fluxnorm_var")
FLUX_TREND_VIEWS = ("flux_trend_global_norm", "flux_trend_global_norm_var")
PERIODOGRAM_VIEWS = ("pgram_smooth_norm", "pgram_tpm_smooth_norm")
CENTROID_VIEWS = ("local_centr_view_std_noclip", "local_centr_view_std_noclip_var")
MOMENTUM_DUMP_VIEWS = ("local_momentum_dump_view", "local_momentum_dump_view_var")
LOCAL_FLUX_VIEWS = ("local_flux_view_fluxnorm", "local_flux_view_fluxnorm_var")
WEAK_SECONDARY_VIEWS = (
    "local_weak_secondary_view_selfnorm",
    "local_weak_secondary_view_selfnorm_var",
)
ODD_VIEWS = ("local_flux_odd_view_fluxnorm", "local_flux_odd_view_fluxnorm_var")
EVEN_VIEWS = ("local_flux_even_view_fluxnorm", "local_flux_even_view_fluxnorm_var")
UNFOLDED_FLUX_VIEW = "unfolded_local_flux_view_fluxnorm"
DIFF_IMG_VIEWS = ("diff_imgs_std_trainset", "oot_imgs_std_trainset", "target_imgs")
QUALITY = "quality"

SCALAR_FEATURES = (
    "boot_fap_norm",
    "flux_even_local_stat_abs_min_norm",
    "flux_global_stat_abs_min_norm",
    "flux_local_stat_abs_min",
    "flux_odd_local_stat_abs_min_norm",
    "flux_trend_global_stat_max_norm",
    "flux_trend_global_stat_min_norm",
    "flux_weak_secondary_local_stat_abs_min_norm",
    "mag_shift_norm",
    "pgram_smooth_max_power_norm",
    "pgram_tpm_smooth_max_power_norm",
    "ruwe_norm",
    "tce_albedo_stat_norm",
    "tce_cap_stat_norm",
    "tce_dikco_msky_err_norm",
    "tce_dikco_msky_norm",
    "tce_hap_stat_norm",
    "tce_max_mult_ev_norm",
    "tce_max_sngle_ev_norm",
    "tce_maxmes_norm",
    "tce_model_chisq_norm",
    "tce_num_transits_norm",
    "tce_num_transits_obs_norm",
    "tce_period_norm",
    "tce_prad_norm",
    "tce_ptemp_stat_norm",
    "tce_robstat_norm",
    "tce_sdens_norm",
    "tce_slogg_norm",
    "tce_smass_norm",
    "tce_smet_norm",
    "tce_sradius_norm",
    "tce_steff_norm",
)

_TRANSIT_VIEWS = (
    CENTROID_VIEWS
    + MOMENTUM_DUMP_VIEWS
    + LOCAL_FLUX_VIEWS
    + WEAK_SECONDARY_VIEWS
    + ODD_VIEWS
    + EVEN_VIEWS
)

FEATURE_SHAPES: dict[str, tuple[int, ...]] = {
    **{name: (FULL_ORBIT_BINS,) for name in GLOBAL_FLUX_VIEWS + FLUX_TREND_VIEWS},
    **{name: (PERIODOGRAM_BINS,) for name in PERIODOGRAM_VIEWS},
    **{name: (TRANSIT_BINS,) for name in _TRANSIT_VIEWS},
    UNFOLDED_FLUX_VIEW: (UNFOLDED_TRANSITS, TRANSIT_BINS),
    **{name: (SECTORS, IMAGE_SIZE, IMAGE_SIZE) for name in DIFF_IMG_VIEWS},
    QUALITY: (SECTORS,),
    **{name: () for name in SCALAR_FEATURES},
}


class UnlabeledBatch(TypedDict):
    """Batch without labels, e.g. unlabeled TCEs to score.

    `features` maps feature names to tensors of shape `(B, *FEATURE_SHAPES[name])`.
    """

    features: dict[str, Tensor]


class Batch(UnlabeledBatch):
    """Labeled batch; `label` has shape `(B,)`, 1.0 for planets (KP/CP), else 0.0."""

    label: Tensor
