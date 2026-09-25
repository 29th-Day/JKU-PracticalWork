import pytest
import torch

from project.features import (
    EVEN_VIEWS,
    ODD_VIEWS,
    UNFOLDED_FLUX_VIEW,
)
from project.inspection import random_inputs
from project.model import DIAGRAM, PAPER, REPO_V1, ExoMinerConfig, ExoMinerPlusPlus
from project.model.branches import (
    DifferenceImageBranch,
    ScalarBranch,
    TimeSeriesBranch,
    TransitViewBranches,
    UnfoldedFluxBranch,
)
from project.model.layers import BranchHead, with_scalars

# Input width of every branch head per preset; derived in spec section 5.2.
HEAD_INPUTS = {
    "PAPER": {
        "global_flux": 4593,
        "flux_trend": 4594,
        "flux_periodogram": 10562,
        "local_centroid": 404,
        "momentum_dump": 400,
        "transit_views.flux": 401,
        "transit_views.weak_secondary": 404,
        "transit_views.odd_even": 402,
        "unfolded_flux": 1602,
        "diff_img": 5,
        "stellar": 6,
        "dv_tce_fit": 9,
    },
    "DIAGRAM": {
        "global_flux": 4593,
        "flux_trend": 4722,
        "flux_periodogram": 10690,
        "local_centroid": 404,
        "momentum_dump": 400,
        "transit_views.flux": 401,
        "transit_views.weak_secondary": 404,
        "transit_views.odd_even": 402,
        "unfolded_flux": 1602,
        "diff_img": 4,
        "stellar": 6,
        "dv_tce_fit": 9,
    },
    "REPO_V1": {
        "global_flux": 4593,
        "flux_trend": 4722,
        "flux_periodogram": 10690,
        "local_centroid": 401,
        "momentum_dump": 400,
        "transit_views.flux": 201,
        "transit_views.weak_secondary": 204,
        "transit_views.odd_even": 202,
        "unfolded_flux": 1602,
        "diff_img": 6,
        "stellar": 7,
        "dv_tce_fit": 9,
    },
}
PRESETS = {"PAPER": PAPER, "DIAGRAM": DIAGRAM, "REPO_V1": REPO_V1}
SINGLE_HEAD_BRANCHES = (
    ScalarBranch,
    TimeSeriesBranch,
    UnfoldedFluxBranch,
    DifferenceImageBranch,
)


def _head_inputs(model: ExoMinerPlusPlus) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name, branch in model.branches.items():
        if isinstance(branch, TransitViewBranches):
            for diagnostic, head in branch.heads.items():
                assert isinstance(head, BranchHead)
                sizes[f"{name}.{diagnostic}"] = head.linear.in_features
        else:
            assert isinstance(branch, SINGLE_HEAD_BRANCHES)
            sizes[name] = branch.head.linear.in_features
    return sizes


@pytest.mark.parametrize("preset", PRESETS)
def test_head_input_sizes_match_spec(preset: str) -> None:
    model = ExoMinerPlusPlus(PRESETS[preset])

    assert _head_inputs(model) == HEAD_INPUTS[preset]


@pytest.mark.parametrize("preset", PRESETS)
def test_branch_outputs_have_branch_units(preset: str) -> None:
    config = PRESETS[preset]
    model = ExoMinerPlusPlus(config)
    inputs = random_inputs(config, 3)

    for name, branch in model.branches.items():
        heads = 3 if name == "transit_views" else 1
        assert branch(inputs).shape == (3, heads * config.branch_units), name


def test_difference_image_full_width_layer_sees_all_sectors() -> None:
    branch = DifferenceImageBranch(PAPER.diff_img, 4, "relu", 0.0)

    assert branch.sectors_layer.in_features == 5 * 16 * 24 * 24 + 5


def test_identical_odd_and_even_views_give_zero_difference() -> None:
    config = PAPER.transit_views
    branch = TransitViewBranches(config, 4, "relu", 0.0).eval()
    inputs = random_inputs(PAPER, 2)
    for odd, even in zip(ODD_VIEWS, EVEN_VIEWS, strict=True):
        inputs[even] = inputs[odd].clone()
    zeros = torch.zeros(2, 16, 25)
    expected = branch.heads["odd_even"](
        with_scalars(zeros, inputs, config.odd_even_scalars)
    )

    assert torch.allclose(branch(inputs)[:, 8:12], expected)


def test_unfolded_branch_ignores_transit_order() -> None:
    branch = UnfoldedFluxBranch(PAPER.unfolded_flux, 4, "relu", 0.0).eval()
    inputs = random_inputs(PAPER, 2)
    shuffled = dict(inputs)
    shuffled[UNFOLDED_FLUX_VIEW] = inputs[UNFOLDED_FLUX_VIEW][:, torch.randperm(20)]

    assert torch.allclose(branch(inputs), branch(shuffled), atol=1e-6)


def test_repo_unfolded_branch_mixes_transits() -> None:
    config: ExoMinerConfig = REPO_V1
    branch = UnfoldedFluxBranch(config.unfolded_flux, 3, "relu", 0.0).eval()
    inputs = random_inputs(config, 2)
    shuffled = dict(inputs)
    shuffled[UNFOLDED_FLUX_VIEW] = inputs[UNFOLDED_FLUX_VIEW][:, torch.randperm(20)]

    assert not torch.allclose(branch(inputs), branch(shuffled), atol=1e-6)
