"""Tests for stability + standardized metrics diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_stability_module():
    module_path = Path(__file__).resolve().parents[1] / "diagnostics" / "run_stability_and_metrics.py"
    spec = importlib.util.spec_from_file_location("run_stability_and_metrics", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load diagnostics/run_stability_and_metrics.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_behavior_csv(tmp_path: Path) -> Path:
    csv_content = """dataset,Hexanol,Benzaldehyde,Linalool,Citral
opto_hex,0.2,0.1,0.3,0.4
"""
    csv_path = tmp_path / "behavior.csv"
    csv_path.write_text(csv_content)
    return csv_path


def _run_stability(module, tmp_path: Path, mock_door_cache: Path, behavior_csv: Path, seed: int) -> Path:
    output_dir = tmp_path
    argv = [
        "run_stability_and_metrics.py",
        "--door_cache",
        str(mock_door_cache),
        "--behavior_csv",
        str(behavior_csv),
        "--conditions",
        "opto_hex",
        "--prediction_mode",
        "test_odorant",
        "--seed",
        str(seed),
        "--output_dir",
        str(output_dir),
        "--lambda_range",
        "1e-4,1e-3,1e-2",
        "--lambda_range_delta",
        "1e-4,1e-3,1e-2",
        "--include_ridge_stability",
    ]
    return output_dir, argv


def test_stability_determinism_and_schema(tmp_path, mock_door_cache, monkeypatch):
    module = _load_stability_module()
    behavior_csv = _write_behavior_csv(tmp_path)

    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"

    run1_dir.mkdir()
    run2_dir.mkdir()

    _, argv1 = _run_stability(module, run1_dir, mock_door_cache, behavior_csv, seed=123)
    monkeypatch.setattr("sys.argv", argv1)
    module.main()

    _, argv2 = _run_stability(module, run2_dir, mock_door_cache, behavior_csv, seed=123)
    monkeypatch.setattr("sys.argv", argv2)
    module.main()

    df1 = pd.read_csv(run1_dir / "stability_per_condition.csv")
    df2 = pd.read_csv(run2_dir / "stability_per_condition.csv")

    pd.testing.assert_frame_equal(df1, df2, check_exact=False, atol=1e-12)

    required_columns = {
        "condition",
        "mode",
        "modelclass",
        "orn_name",
        "selection_frequency",
        "sign_consistency",
        "mean_abs_weight",
        "std_abs_weight",
        "mean_rank",
        "n_folds",
    }
    assert required_columns.issubset(df1.columns)

    metrics_df = pd.read_csv(run1_dir / "model_metrics.csv")
    metrics_required = {
        "condition",
        "mode",
        "modelclass",
        "cv_mse",
        "nmse",
        "rmse_over_y_std",
        "intercept_only_flag",
        "intercept_only_mse",
        "y_var",
        "y_min",
        "y_max",
        "pred_min",
        "pred_max",
    }
    assert metrics_required.issubset(metrics_df.columns)


def test_intercept_only_flag_logic():
    module = _load_stability_module()
    assert module._is_intercept_only(0, 0.0, 1.0, 1.0)
    assert not module._is_intercept_only(1, 0.0, 1.0, 1.0)
    assert not module._is_intercept_only(0, 1e-3, 1.0, 1.0)
    assert not module._is_intercept_only(0, 0.0, 1.0, 1.1)
