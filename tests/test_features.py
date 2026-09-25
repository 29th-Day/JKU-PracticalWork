from project.features import (
    CENTROID_VIEWS,
    DIFF_IMG_VIEWS,
    EVEN_VIEWS,
    FEATURE_SHAPES,
    FLUX_TREND_VIEWS,
    GLOBAL_FLUX_VIEWS,
    LOCAL_FLUX_VIEWS,
    MOMENTUM_DUMP_VIEWS,
    ODD_VIEWS,
    PERIODOGRAM_VIEWS,
    QUALITY,
    SCALAR_FEATURES,
    UNFOLDED_FLUX_VIEW,
    WEAK_SECONDARY_VIEWS,
)


def test_view_shapes() -> None:
    full_orbit = GLOBAL_FLUX_VIEWS + FLUX_TREND_VIEWS
    transit = (
        CENTROID_VIEWS
        + MOMENTUM_DUMP_VIEWS
        + LOCAL_FLUX_VIEWS
        + WEAK_SECONDARY_VIEWS
        + ODD_VIEWS
        + EVEN_VIEWS
    )

    assert all(FEATURE_SHAPES[name] == (301,) for name in full_orbit)
    assert all(FEATURE_SHAPES[name] == (674,) for name in PERIODOGRAM_VIEWS)
    assert all(FEATURE_SHAPES[name] == (31,) for name in transit)
    assert FEATURE_SHAPES[UNFOLDED_FLUX_VIEW] == (20, 31)
    assert all(FEATURE_SHAPES[name] == (5, 33, 33) for name in DIFF_IMG_VIEWS)
    assert FEATURE_SHAPES[QUALITY] == (5,)


def test_scalars_are_scalar_shaped() -> None:
    assert len(SCALAR_FEATURES) == len(set(SCALAR_FEATURES)) == 33
    assert all(FEATURE_SHAPES[name] == () for name in SCALAR_FEATURES)


def test_feature_shapes_has_no_unlisted_features() -> None:
    views = (
        GLOBAL_FLUX_VIEWS
        + FLUX_TREND_VIEWS
        + PERIODOGRAM_VIEWS
        + CENTROID_VIEWS
        + MOMENTUM_DUMP_VIEWS
        + LOCAL_FLUX_VIEWS
        + WEAK_SECONDARY_VIEWS
        + ODD_VIEWS
        + EVEN_VIEWS
        + DIFF_IMG_VIEWS
    )
    listed = {*views, UNFOLDED_FLUX_VIEW, QUALITY, *SCALAR_FEATURES}

    assert set(FEATURE_SHAPES) == listed
