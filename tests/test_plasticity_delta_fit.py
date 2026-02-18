"""Tests for plasticity delta-weight fitting pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd

from door_toolkit.plasticity_delta_fit import (
    build_reconstruction_table,
    build_weight_comparison_table,
    compute_delta,
    export_reports,
    extract_four_odors,
    filter_training_conditions,
    fit_delta_weights_per_condition,
    load_baseline_weights,
    load_per_matrix,
    mean_center_rows,
)


FIXTURE_CSV = Path(__file__).parent / "data" / "plasticity_delta_fixture.csv"
BASELINE_FIXTURE_DIR = Path(__file__).parent / "data" / "baseline_fixture"
DELTA_WEIGHTS_FIXTURE = Path(__file__).parent / "data" / "delta_weights_fixture.csv"


def _mock_feature_builder(odors):
    """Deterministic 4-odor feature matrix for stable sparse fits in tests."""
    assert list(odors) == ["benzaldehyde", "ethyl_butyrate", "1-hexanol", "3-octanol"]
    X = np.array(
        [
            [1.0, 0.0, 0.0],  # benz
            [0.0, 1.0, 0.0],  # EB
            [1.0, 0.0, 1.0],  # hex
            [0.0, 1.0, 1.0],  # 3-oct
        ],
        dtype=np.float64,
    )
    feature_names = ["f_benz_like", "f_eb_like", "f_shared"]
    return X, feature_names, {"source": "unit_test"}


def test_load_per_matrix_resolves_columns_and_conditions():
    df, odor_colmap, conditions = load_per_matrix(str(FIXTURE_CSV), control_row="opto_AIR")

    assert "opto_AIR" in df.index
    assert list(odor_colmap.keys()) == [
        "benzaldehyde",
        "ethyl_butyrate",
        "1-hexanol",
        "3-octanol",
    ]
    assert odor_colmap == {
        "benzaldehyde": "Benzaldehyde",
        "ethyl_butyrate": "Ethyl_Butyrate",
        "1-hexanol": "Hexanol",
        "3-octanol": "3-Octonol",
    }
    # Non-opto rows are excluded from inferred training conditions.
    assert conditions == ["opto_benz_1", "opto_hex"]


def test_compute_delta_uses_opto_air_row_not_hardcoded():
    df, odor_colmap, conditions = load_per_matrix(str(FIXTURE_CSV), control_row="opto_AIR")
    df4 = extract_four_odors(df.loc[["opto_AIR"] + conditions], odor_colmap)

    delta = compute_delta(df4, control_row="opto_AIR", trained_conditions=conditions)

    expected_benz = pd.Series(
        {
            "benzaldehyde": 0.10,
            "ethyl_butyrate": -0.10,
            "1-hexanol": 0.10,
            "3-octanol": -0.10,
        }
    )
    expected_hex = pd.Series(
        {
            "benzaldehyde": -0.10,
            "ethyl_butyrate": 0.05,
            "1-hexanol": 0.20,
            "3-octanol": -0.05,
        }
    )
    pd.testing.assert_series_equal(delta.loc["opto_benz_1"], expected_benz, check_names=False)
    pd.testing.assert_series_equal(delta.loc["opto_hex"], expected_hex, check_names=False)


def test_mean_center_rows_sets_each_condition_mean_to_zero():
    df, odor_colmap, conditions = load_per_matrix(str(FIXTURE_CSV), control_row="opto_AIR")
    df4 = extract_four_odors(df.loc[["opto_AIR"] + conditions], odor_colmap)
    delta = compute_delta(df4, control_row="opto_AIR", trained_conditions=conditions)

    centered = mean_center_rows(delta)

    assert np.allclose(centered.mean(axis=1).to_numpy(dtype=float), 0.0, atol=1e-12)


def test_sparse_fit_is_deterministic_and_has_stable_signs():
    df, odor_colmap, conditions = load_per_matrix(str(FIXTURE_CSV), control_row="opto_AIR")
    df4 = extract_four_odors(df.loc[["opto_AIR"] + conditions], odor_colmap)
    delta = compute_delta(df4, control_row="opto_AIR", trained_conditions=conditions)
    centered = mean_center_rows(delta)

    fit1 = fit_delta_weights_per_condition(
        odors=list(centered.columns),
        y_centered=centered,
        feature_builder=_mock_feature_builder,
        model="lasso",
        alpha_grid=[1e-4, 1e-3, 1e-2],
        random_state=0,
        standardize=True,
        zero_eps=1e-6,
    )
    fit2 = fit_delta_weights_per_condition(
        odors=list(centered.columns),
        y_centered=centered,
        feature_builder=_mock_feature_builder,
        model="lasso",
        alpha_grid=[1e-4, 1e-3, 1e-2],
        random_state=0,
        standardize=True,
        zero_eps=1e-6,
    )

    for condition in conditions:
        r1 = fit1["condition_results"][condition]
        r2 = fit2["condition_results"][condition]
        np.testing.assert_allclose(r1["weights"], r2["weights"])
        np.testing.assert_array_equal(r1["weight_signs"], r2["weight_signs"])
        assert r1["alpha"] == r2["alpha"]
        assert set(r1["weight_signs"]).issubset({-1, 0, 1})


def test_export_reports_writes_benz_to_hex_contrib_file(tmp_path):
    df, odor_colmap, conditions = load_per_matrix(str(FIXTURE_CSV), control_row="opto_AIR")
    df4 = extract_four_odors(df.loc[["opto_AIR"] + conditions], odor_colmap)
    delta = compute_delta(df4, control_row="opto_AIR", trained_conditions=conditions)
    centered = mean_center_rows(delta)
    fit_results = fit_delta_weights_per_condition(
        odors=list(centered.columns),
        y_centered=centered,
        feature_builder=_mock_feature_builder,
        model="lasso",
        alpha_grid=[1e-4, 1e-3, 1e-2],
        random_state=0,
    )

    exported = export_reports(
        str(tmp_path),
        df4=df4,
        control_row="opto_AIR",
        odor_colmap=odor_colmap,
        delta_raw=delta,
        delta_centered=centered,
        fit_results=fit_results,
    )

    assert exported["delta_weights"]
    assert exported["contributions"]

    benz_to_hex = tmp_path / "contrib_opto_benz_1_hexanol.csv"
    assert benz_to_hex.exists()

    weights_df = pd.read_csv(exported["delta_weights"][0])
    assert "sign" in weights_df.columns
    assert set(weights_df["sign"].unique()).issubset({"+", "-", "0"})


def test_condition_filtering_allow_and_block():
    df, _odor_colmap, _ = load_per_matrix(str(FIXTURE_CSV), control_row="opto_AIR")
    allow_only = filter_training_conditions(
        df,
        control_row="opto_AIR",
        allowlist=["opto_hex"],
        blocklist=None,
    )
    assert allow_only == ["opto_hex"]

    block_hex = filter_training_conditions(
        df,
        control_row="opto_AIR",
        allowlist=None,
        blocklist=["opto_hex"],
    )
    assert block_hex == ["opto_benz_1"]


def test_weight_comparison_table_merging():
    baseline_df, _summary = load_baseline_weights(str(BASELINE_FIXTURE_DIR))
    delta_df = pd.read_csv(DELTA_WEIGHTS_FIXTURE)
    comparison = build_weight_comparison_table(
        baseline_df=baseline_df,
        delta_features=delta_df["feature"].tolist(),
        delta_weights=delta_df["delta_weight"].to_numpy(),
        zero_eps=1e-6,
    )
    assert set(
        [
            "feature",
            "baseline_w",
            "delta_w",
            "combined_w",
            "baseline_sign",
            "delta_sign",
            "combined_sign",
            "abs_delta",
            "rank_baseline",
            "rank_delta",
            "rank_combined",
            "rank_change",
        ]
    ).issubset(comparison.columns)
    assert comparison["combined_w"].iloc[0] == comparison["baseline_w"].iloc[0] + comparison["delta_w"].iloc[0]


def test_reconstruction_table_centered_addback():
    df, odor_colmap, conditions = load_per_matrix(str(FIXTURE_CSV), control_row="opto_AIR")
    df4 = extract_four_odors(df.loc[["opto_AIR"] + conditions], odor_colmap)
    delta_raw = compute_delta(df4, control_row="opto_AIR", trained_conditions=conditions)
    delta_centered = mean_center_rows(delta_raw)

    condition = "opto_benz_1"
    y_pred_full = delta_centered.loc[condition].to_numpy(dtype=float)
    fit_results = {
        "odors": list(df4.columns),
        "condition_results": {condition: {"y_pred_full": y_pred_full}},
    }
    recon = build_reconstruction_table(
        df4=df4,
        delta_raw=delta_raw,
        delta_centered=delta_centered,
        fit_results=fit_results,
        condition_name=condition,
        control_row="opto_AIR",
        centered=True,
    )
    delta_mean = float(delta_raw.loc[condition].mean())
    np.testing.assert_allclose(
        recon["delta_per_pred_raw"].to_numpy(),
        recon["delta_per_pred"].to_numpy() + delta_mean,
    )
    np.testing.assert_allclose(
        recon["trained_per_pred"].to_numpy(),
        recon["control_per_true"].to_numpy() + recon["delta_per_pred_raw"].to_numpy(),
    )
