#!/usr/bin/env python3
"""
Run LASSO behavioral prediction with optional control subtraction.

Example:
  python scripts/run_lasso_behavioral_prediction.py \
    --door_cache door_cache \
    --behavior_csv /path/to/reaction_rates_summary_unordered.csv \
    --condition opto_hex \
    --output_dir outputs/lasso_behavioral_prediction \
    --prediction_mode test_odorant \
    --cv_folds 5 \
    --lambda_range 0.0001,0.001,0.01,0.1,1.0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

# Add src to path for repo-local runs (matches other scripts in this repo).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from door_toolkit.pathways import LassoBehavioralPredictor

logger = logging.getLogger(__name__)


def _parse_conditions(values: List[str]) -> List[str]:
    conditions: List[str] = []
    seen = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token or token in seen:
                continue
            conditions.append(token)
            seen.add(token)
    return conditions


def _parse_lambda_range(value: Optional[str]) -> Optional[List[float]]:
    if value is None:
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("lambda_range cannot be empty.")
    try:
        return [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"Invalid lambda_range value: {value}") from exc


def _run_condition(
    predictor: LassoBehavioralPredictor,
    *,
    condition_name: str,
    mode_label: str,
    subtract_control: bool,
    control_condition: Optional[str],
    missing_control_policy: str,
    prediction_mode: str,
    lambda_range: Optional[List[float]],
    cv_folds: int,
    output_dir: Path,
):
    logger.info("Running %s mode for condition '%s'", mode_label, condition_name)
    condition_dir = output_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)
    results = predictor.fit_behavior(
        condition_name=condition_name,
        lambda_range=lambda_range,
        cv_folds=cv_folds,
        prediction_mode=prediction_mode,
        subtract_control=subtract_control,
        control_condition=control_condition,
        missing_control_policy=missing_control_policy,
    )

    prefix = f"{mode_label}"
    results.plot_predictions(save_to=str(condition_dir / f"{prefix}_predictions.png"))
    results.plot_receptors(save_to=str(condition_dir / f"{prefix}_receptors.png"))
    results.export_csv(str(condition_dir / f"{prefix}_results.csv"))
    results.export_json(str(condition_dir / f"{prefix}_model.json"))

    return results


def _is_missing_control_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "no matched control mapping" in message:
        return True
    if "control" in message and "not found in behavioral data" in message:
        return True
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LASSO behavioral prediction with optional control subtraction.",
    )

    parser.add_argument("--door_cache", required=True, help="Path to DoOR cache directory.")
    parser.add_argument("--behavior_csv", required=True, help="Path to behavioral matrix CSV.")
    parser.add_argument(
        "--condition",
        required=True,
        action="append",
        help="Condition name(s). Repeat flag or pass comma-separated list.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to write outputs.")

    parser.add_argument("--subtract_control", action="store_true", help="Fit ΔPER = opto - control.")
    parser.add_argument(
        "--also_run_raw",
        action="store_true",
        help="If set, run raw PER alongside ΔPER and write a comparison summary.",
    )
    parser.add_argument(
        "--control_condition",
        default=None,
        help="Optional control dataset override (applies to all conditions).",
    )
    parser.add_argument(
        "--missing_control_policy",
        choices=["skip", "zero", "error"],
        default="skip",
        help="How to handle missing control values.",
    )

    parser.add_argument(
        "--prediction_mode",
        choices=["test_odorant", "trained_odorant", "interaction"],
        default="test_odorant",
        help="Feature extraction mode.",
    )
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument(
        "--lambda_range",
        default=None,
        help="Comma-separated lambda values (e.g., 0.0001,0.001,0.01).",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()

    conditions = _parse_conditions(args.condition)
    if not conditions:
        raise ValueError("No valid conditions provided.")

    lambda_range = _parse_lambda_range(args.lambda_range)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = LassoBehavioralPredictor(
        doorcache_path=args.door_cache,
        behavior_csv_path=args.behavior_csv,
    )

    modes_to_run = []
    if args.also_run_raw or not args.subtract_control:
        modes_to_run.append(("raw", False))
    if args.subtract_control:
        modes_to_run.append(("delta", True))

    summary_rows = []
    for condition_name in conditions:
        ran_raw = False
        for mode_label, subtract_control in modes_to_run:
            try:
                results = _run_condition(
                    predictor,
                    condition_name=condition_name,
                    mode_label=mode_label,
                    subtract_control=subtract_control,
                    control_condition=args.control_condition,
                    missing_control_policy=args.missing_control_policy,
                    prediction_mode=args.prediction_mode,
                    lambda_range=lambda_range,
                    cv_folds=args.cv_folds,
                    output_dir=output_dir,
                )
            except ValueError as exc:
                if subtract_control and _is_missing_control_error(exc):
                    logger.warning(
                        "No control found for '%s'; falling back to raw mode.",
                        condition_name,
                    )
                    if ran_raw:
                        logger.info("Raw mode already completed for '%s'; skipping delta.", condition_name)
                        continue
                    results = _run_condition(
                        predictor,
                        condition_name=condition_name,
                        mode_label="raw",
                        subtract_control=False,
                        control_condition=None,
                        missing_control_policy=args.missing_control_policy,
                        prediction_mode=args.prediction_mode,
                        lambda_range=lambda_range,
                        cv_folds=args.cv_folds,
                        output_dir=output_dir,
                    )
                    mode_label = "raw"
                else:
                    raise

            if mode_label == "raw":
                ran_raw = True

            summary_rows.append(
                {
                    "condition": condition_name,
                    "mode": mode_label,
                    "cv_mse": results.cv_mse,
                    "cv_r2": results.cv_r2_score,
                    "n_receptors_selected": results.n_receptors_selected,
                    "control_condition": results.control_condition,
                }
            )

    if args.also_run_raw:
        import pandas as pd

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "lasso_summary_comparison.csv", index=False)
        logger.info("Wrote comparison summary to %s", output_dir / "lasso_summary_comparison.csv")


if __name__ == "__main__":
    main()
