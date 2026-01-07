#!/usr/bin/env python3
"""
Train dataset-level PER rate model (Option 1).

This script fits a sparse GLM that predicts PER rates for each (dataset, test odor)
cell in a behavioral matrix (rows = datasets/conditions, columns = test odors).

Example:
  python scripts/train_behavior_rate_model.py \\
    --behavior_csv /path/to/reaction_rates_summary_unordered.csv \\
    --output_dir outputs/behavior_rate_model \\
    --epochs 500 --lr 1e-2 --l1_lambda 1e-3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add src to path for repo-local runs (matches other scripts in this repo).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from door_toolkit.encoder import DoOREncoder
from door_toolkit.pathways.behavior_rate_model import (
    OdorResolutionLogger,
    TrainConfig,
    ablation_importance,
    build_adult_receptor_mask,
    build_prediction_long_format,
    build_training_table,
    compute_learning_effect_metrics,
    compute_per_dataset_metrics,
    compute_per_test_odor_metrics,
    default_dataset_trained_odor_mapping,
    default_behavior_odorant_candidates,
    evaluate_fit,
    load_behavior_matrix_csv,
    predict_rates,
    resolve_door_odor_name,
    resolve_trained_odor_for_dataset,
    save_json,
    split_observed_extrapolated,
    train_sparse_rate_glm,
    weight_importance_dataframe,
)

logger = logging.getLogger(__name__)


def _git_sha() -> Optional[str]:
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a dataset-level PER rate GLM from DoOR receptor profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Notes:\n"
            "- Training uses only observed (non-NaN) cells.\n"
            "- By default, trains on datasets starting with 'opto_' and excludes 'opto_AIR'.\n"
        ),
    )

    parser.add_argument("--behavior_csv", required=True, help="Path to behavioral matrix CSV.")
    parser.add_argument(
        "--door_cache",
        default="door_cache",
        help="Path to DoOR cache directory (default: door_cache).",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to write artifacts.")

    # Dataset selection
    parser.add_argument(
        "--train_prefix",
        default="opto_",
        help="Train only on datasets with this prefix (default: opto_). Use '' to disable.",
    )
    parser.add_argument(
        "--exclude_dataset",
        action="append",
        default=["opto_AIR"],
        help="Dataset(s) to exclude from training (default: opto_AIR). Can repeat.",
    )

    # Model config
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--l1_lambda", type=float, default=1e-3)

    parser.add_argument("--no_test_block", action="store_true", help="Disable w_test·x_test block.")
    parser.add_argument("--no_train_block", action="store_true", help="Disable w_train·x_train block.")
    parser.add_argument(
        "--no_interaction_block",
        action="store_true",
        help="Disable w_int·(x_test ⊙ x_train) block.",
    )
    parser.add_argument("--diff_block", action="store_true", help="Enable w_diff·(x_test-x_train) block.")
    parser.add_argument(
        "--no_reward_term",
        action="store_true",
        help="Disable reward_flag term (auto-disabled if constant in training).",
    )

    # Mappings
    parser.add_argument(
        "--trained_odor_map_json",
        default=None,
        help="Optional JSON mapping: dataset -> trained_odor_label (overrides defaults).",
    )

    # Importance / ablations
    parser.add_argument(
        "--top_k_ablation",
        type=int,
        default=15,
        help="Top-K ORNs to include in ablation_sweep.csv (default: 15).",
    )

    # Optional connectome-aware interpretation (post-training; does not change training objective)
    parser.add_argument(
        "--connectome_dir",
        default=None,
        help=(
            "Optional directory containing FlyWire connectivity artifacts: "
            "orn_pn_connectivity.pt, pn_kc_connectivity.pt, connectivity_metadata.json. "
            "If omitted, auto-detect under repo data/pgcn_features/connectivity or data/connectivity."
        ),
    )
    parser.add_argument(
        "--disable_connectome_analysis",
        action="store_true",
        help="Disable post-training connectome analysis even if connectivity files are available.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail on connectome alignment errors instead of continuing with warning. "
            "Recommended for audit/validation runs to catch silent data misalignment."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    # Load behavioral CSV with validation
    behavior_df, format_meta = load_behavior_matrix_csv(args.behavior_csv, validate_rates=True)
    logger.info(
        "Loaded behavior CSV: %s format, %d datasets, %d odors",
        format_meta["format_type"],
        format_meta["n_datasets"],
        format_meta["n_odors"],
    )
    if format_meta.get("rescaled"):
        logger.warning(
            "Rescaled rates from [0,100] to [0,1] for %d cells",
            format_meta["n_cells"],
        )
    save_json(out_dir / "input_format_metadata.json", format_meta)

    encoder = DoOREncoder(args.door_cache, use_torch=False)

    # Initialize resolution logger
    resolution_logger = OdorResolutionLogger()

    # Load adult receptor mask
    receptor_mask, mask_manifest = build_adult_receptor_mask(
        encoder,
        training_receptor_set_path=Path(__file__).parents[1] / "data/mappings/training_receptor_set.json",
    )
    save_json(out_dir / "adult_only_mask_manifest.json", mask_manifest)
    logger.info(
        "Receptor mask: %d adult ORNs selected from %d total",
        mask_manifest["n_adult_only"],
        mask_manifest["n_total_receptors"],
    )

    odorant_candidates = default_behavior_odorant_candidates()
    dataset_trained_map: Dict[str, str] = default_dataset_trained_odor_mapping()
    if args.trained_odor_map_json:
        user_map = json.loads(Path(args.trained_odor_map_json).read_text(encoding="utf-8"))
        if not isinstance(user_map, dict):
            raise ValueError("--trained_odor_map_json must contain a JSON object mapping dataset -> odor label.")
        dataset_trained_map.update({str(k): str(v) for k, v in user_map.items()})

    # Select training datasets
    include_datasets: Optional[List[str]] = None
    if str(args.train_prefix) != "":
        include_datasets = [d for d in behavior_df.index if str(d).startswith(str(args.train_prefix))]
    exclude_datasets = list(args.exclude_dataset) if args.exclude_dataset else []

    train_table = build_training_table(
        behavior_df,
        encoder,
        include_datasets=include_datasets,
        exclude_datasets=exclude_datasets,
        odorant_candidates=odorant_candidates,
        dataset_trained_odor_map=dataset_trained_map,
        resolution_logger=resolution_logger,
        receptor_mask=receptor_mask,
    )

    # Save resolution log
    save_json(out_dir / "odor_name_resolution.json", resolution_logger.to_json())

    config = TrainConfig(
        seed=int(args.seed),
        epochs=int(args.epochs),
        lr=float(args.lr),
        l1_lambda=float(args.l1_lambda),
        include_test=not bool(args.no_test_block),
        include_train=not bool(args.no_train_block),
        include_interaction=not bool(args.no_interaction_block),
        include_diff=bool(args.diff_block),
        include_reward=not bool(args.no_reward_term),
    )

    logger.info("Training rows: %d observed cells", len(train_table.y))
    logger.info("Receptor channels: %d", len(train_table.receptor_names))

    model, loss_history = train_sparse_rate_glm(train_table, config)
    fit = evaluate_fit(model, train_table, l1_lambda=config.l1_lambda)

    # Predict full matrix (all datasets × all test odors)
    datasets = list(behavior_df.index.astype(str))
    odors = [str(c) for c in behavior_df.columns]

    # Determine effective number of channels (after masking)
    n_effective_channels = receptor_mask.sum() if receptor_mask is not None else encoder.n_channels

    # Resolve test odor vectors
    test_vecs: List[np.ndarray] = []
    unresolved_test: List[str] = []
    for odor in odors:
        door_name = resolve_door_odor_name(odor, encoder, odorant_candidates=odorant_candidates)
        if door_name is None:
            # Includes AIR and unknown odorants: encode as zeros.
            if str(odor).strip().lower() != "air":
                unresolved_test.append(odor)
            vec = np.zeros(n_effective_channels, dtype=np.float32)
        else:
            vec = np.asarray(encoder.encode(door_name, fill_missing=0.0), dtype=np.float32)
            # Apply receptor mask if provided
            if receptor_mask is not None:
                vec = vec[receptor_mask]
        test_vecs.append(vec)

    # Resolve trained odor vectors per dataset
    train_vecs: List[np.ndarray] = []
    unresolved_datasets: List[str] = []
    reward_flags: List[float] = []
    for ds in datasets:
        try:
            trained_label = resolve_trained_odor_for_dataset(ds, mapping=dataset_trained_map)
        except ValueError:
            unresolved_datasets.append(ds)
            raise
        door_name = resolve_door_odor_name(trained_label, encoder, odorant_candidates=odorant_candidates)
        if door_name is None:
            vec = np.zeros(n_effective_channels, dtype=np.float32)
        else:
            vec = np.asarray(encoder.encode(door_name, fill_missing=0.0), dtype=np.float32)
            # Apply receptor mask if provided
            if receptor_mask is not None:
                vec = vec[receptor_mask]
        train_vecs.append(vec)
        reward_flags.append(1.0 if ds.startswith("opto_") else 0.0)

    x_test = np.stack(test_vecs, axis=0)  # (O, C_effective)
    x_train = np.stack(train_vecs, axis=0)  # (D, C_effective)
    reward = np.asarray(reward_flags, dtype=np.float32)  # (D,)

    # Build grid tensors: (D*O, C)
    import torch

    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    reward_t = torch.tensor(reward, dtype=torch.float32)

    x_test_grid = x_test_t.unsqueeze(0).expand(len(datasets), len(odors), -1).reshape(-1, x_test_t.shape[1])
    x_train_grid = x_train_t.unsqueeze(1).expand(len(datasets), len(odors), -1).reshape(-1, x_train_t.shape[1])
    reward_grid = reward_t.repeat_interleave(len(odors))

    p_hat = predict_rates(model, x_test_grid, x_train_grid, reward_grid).reshape(len(datasets), len(odors))
    pred_df = pd.DataFrame(p_hat.detach().cpu().numpy(), index=datasets, columns=odors)

    # Convert to long format with metadata
    predictions_long = build_prediction_long_format(
        behavior_df,
        pred_df,
        dataset_trained_map,
        resolution_logger,
    )

    # Split observed vs extrapolated
    observed_df, extrapolated_df = split_observed_extrapolated(predictions_long)

    # Save long-format outputs
    observed_df.to_csv(out_dir / "predicted_rates.csv", index=False)
    extrapolated_df.to_csv(out_dir / "extrapolated_predictions.csv", index=False)

    logger.info(
        "Wrote %d observed cells, %d extrapolated cells",
        len(observed_df),
        len(extrapolated_df),
    )

    # Per-dataset metrics
    per_dataset = compute_per_dataset_metrics(observed_df)
    per_dataset.to_csv(out_dir / "metrics_per_dataset.csv", index=False)

    # Per-test-odor metrics
    per_odor = compute_per_test_odor_metrics(observed_df)
    per_odor.to_csv(out_dir / "metrics_per_test_odor.csv", index=False)

    # Matched opto→control pairs (used for "learning effect" summaries when available)
    opto_control_pairs: Dict[str, str] = {
        "opto_hex": "hex_control",
        "opto_EB": "EB_control",
        "opto_EB_6_training": "EB_control",
        "opto_benz_1": "Benz_control",
    }

    available_pairs = {
        opto: ctrl for opto, ctrl in opto_control_pairs.items() if opto in pred_df.index and ctrl in pred_df.index
    }
    if available_pairs:
        # Compute learning effect metrics
        learning_effect = compute_learning_effect_metrics(predictions_long, available_pairs)
        learning_effect.to_csv(out_dir / "learning_effect_metrics.csv", index=False)

        # Also save old-format for backward compatibility (can be removed later)
        effect_df = pd.DataFrame(
            {opto: (pred_df.loc[opto] - pred_df.loc[ctrl]) for opto, ctrl in available_pairs.items()}
        ).T
        effect_df.index.name = "opto_dataset"
        effect_df.to_csv(out_dir / "predicted_learning_effect_opto_minus_control.csv")
    else:
        effect_df = None

    # Importance
    w_df = weight_importance_dataframe(model, train_table.receptor_names)
    ab_global, ab_by_dataset, ab_by_odor = ablation_importance(model, train_table)

    orn_global = pd.merge(w_df, ab_global, on="receptor", how="left")
    orn_global.to_csv(out_dir / "orn_importance_global.csv", index=False)
    ab_by_dataset.to_csv(out_dir / "orn_importance_by_dataset.csv", index=False)
    ab_by_odor.to_csv(out_dir / "orn_importance_by_test_odor.csv", index=False)

    # Ablation sweep (top-K by ΔBCE)
    top_k = max(1, int(args.top_k_ablation))
    ab_sweep = ab_global.head(top_k).copy().reset_index(drop=True)
    ab_sweep.insert(0, "rank", np.arange(1, len(ab_sweep) + 1))

    # Effect deltas (optional): recompute predictions with ablated channels for top-K.
    if effect_df is not None:
        effect_base = effect_df.to_numpy(dtype=np.float64)
        n_effect_cells = int(effect_base.size)
        receptor_to_index = {r: i for i, r in enumerate(train_table.receptor_names)}

        delta_effect_mean: List[float] = []
        delta_effect_abs_mean: List[float] = []

        for receptor in ab_sweep["receptor"].tolist():
            i = receptor_to_index[receptor]
            x_test_ab = x_test_grid.clone()
            x_train_ab = x_train_grid.clone()
            x_test_ab[:, i] = 0.0
            x_train_ab[:, i] = 0.0

            p_ab = predict_rates(model, x_test_ab, x_train_ab, reward_grid).reshape(len(datasets), len(odors))
            pred_ab_df = pd.DataFrame(p_ab.detach().cpu().numpy(), index=datasets, columns=odors)
            eff_ab = pd.DataFrame(
                {opto: (pred_ab_df.loc[opto] - pred_ab_df.loc[ctrl]) for opto, ctrl in available_pairs.items()}
            ).T.to_numpy(dtype=np.float64)

            delta = eff_ab - effect_base
            delta_effect_mean.append(float(np.mean(delta)))
            delta_effect_abs_mean.append(float(np.mean(np.abs(eff_ab) - np.abs(effect_base))))

        ab_sweep["n_effect_cells"] = n_effect_cells
        ab_sweep["delta_effect_mean"] = delta_effect_mean
        ab_sweep["delta_effect_abs_mean"] = delta_effect_abs_mean
    else:
        ab_sweep["n_effect_cells"] = 0
        ab_sweep["delta_effect_mean"] = np.nan
        ab_sweep["delta_effect_abs_mean"] = np.nan

    ab_sweep.to_csv(out_dir / "ablation_sweep.csv", index=False)

    # Connectome-aware post-training analysis (optional; safe-default).
    if not bool(args.disable_connectome_analysis):
        from door_toolkit.pathways.connectome_analysis import (
            align_orn_connectome,
            autodetect_connectome_dir,
            build_orn_to_pn_top_edges,
            compute_connectome_influence,
            load_connectome_matrices,
            orient_connectome,
            write_connectome_outputs,
        )

        repo_root = Path(__file__).resolve().parents[1]
        user_connectome_dir = Path(args.connectome_dir) if args.connectome_dir else None
        connectome_dir = user_connectome_dir or autodetect_connectome_dir(repo_root)

        if connectome_dir is None:
            if args.strict:
                raise FileNotFoundError(
                    "Connectome analysis failed (--strict mode): "
                    "No connectivity artifacts found. Provide --connectome_dir or disable --strict."
                )
            logger.info("Connectome analysis skipped (no connectivity artifacts found).")
        else:
            # Enable strict mode if user explicitly provided --connectome_dir OR --strict flag
            strict = (user_connectome_dir is not None) or bool(args.strict)
            if strict:
                logger.info("Connectome analysis: strict mode enabled (errors will raise)")
            try:
                A_raw, B_raw, meta = load_connectome_matrices(connectome_dir)
                connectome_receptors = meta.get("receptor_names") or meta.get("connectivity_receptor_order")
                if not isinstance(connectome_receptors, list) or not all(
                    isinstance(x, str) for x in connectome_receptors
                ):
                    raise ValueError(
                        "connectivity_metadata.json must provide a string list 'receptor_names' to align ORN ordering."
                    )

                # Align base importance (model order) to connectome ORN axis via names.
                n_channels = len(train_table.receptor_names)

                def _to_np(param) -> np.ndarray:
                    if param is None:
                        return np.zeros(n_channels, dtype=np.float64)
                    return param.detach().cpu().numpy().astype(np.float64)

                w_test = _to_np(model.w_test)
                w_train = _to_np(model.w_train)
                w_int = _to_np(model.w_int)
                w_diff = _to_np(model.w_diff)
                base_importance = np.abs(w_test) + np.abs(w_train) + np.abs(w_int) + np.abs(w_diff)

                A_pn_by_orn, B_kc_by_pn, orientation_report = orient_connectome(
                    A_raw, B_raw, n_orn=n_channels
                )
                A_aligned, alignment_report = align_orn_connectome(
                    A_pn_by_orn, connectome_receptors, train_table.receptor_names
                )

                pn_ids = meta.get("pn_root_ids")
                kc_ids = meta.get("kc_root_ids")

                influence = compute_connectome_influence(
                    base_importance,
                    A_aligned,
                    B_kc_by_pn,
                    receptor_names=train_table.receptor_names,
                    pn_ids=pn_ids,
                    kc_ids=kc_ids,
                    top_pn=100,
                    top_kc=200,
                )

                top_orns = influence.orn_table["receptor"].head(15).tolist()
                orn_to_pn_edges = build_orn_to_pn_top_edges(
                    A_aligned,
                    receptor_names=train_table.receptor_names,
                    pn_ids=pn_ids,
                    top_orns=top_orns,
                    top_edges_per_orn=30,
                )

                connectome_inputs = {
                    "seed": int(args.seed),
                    "base_importance_definition": "abs(w_test)+abs(w_train)+abs(w_int)+abs(w_diff)",
                    "model_receptor_order": list(train_table.receptor_names),
                    "connectome": {
                        "paths": meta.get("_paths", {}),
                        "hashes": meta.get("_hashes", {}),
                        "loaded_shapes": meta.get("_loaded_shapes", {}),
                        "metadata": {
                            "n_receptors": meta.get("n_receptors"),
                            "n_pns": meta.get("n_pns"),
                            "n_kcs": meta.get("n_kcs"),
                        },
                    },
                    "orientation": orientation_report,
                    "alignment": alignment_report,
                    "notes": {
                        "strict_mode": bool(strict),
                        "if_strict_true_then_errors_raise": True,
                    },
                }

                write_connectome_outputs(
                    out_dir,
                    {
                        "connectome_inputs": connectome_inputs,
                        "orn_table": influence.orn_table,
                        "pn_table": influence.pn_table,
                        "kc_table": influence.kc_table,
                        "summary": influence.summary,
                        "orn_to_pn_edges": orn_to_pn_edges,
                    },
                )
                logger.info(
                    "Connectome alignment successful: ORN×PN=%s, PN×KC=%s",
                    A_aligned.shape,
                    B_kc_by_pn.shape,
                )
                logger.info("Wrote: %s", out_dir / "connectome_analysis")
            except Exception as e:
                if strict:
                    logger.error("Connectome alignment failed (--strict mode): %s", e)
                    raise
                logger.warning("Connectome analysis skipped due to error: %s", e)

    # Metrics + run config
    metrics = {
        "n_datasets_total": int(behavior_df.shape[0]),
        "n_test_odors_total": int(behavior_df.shape[1]),
        "n_train_rows": int(len(train_table.y)),
        "fit": {
            "bce": fit.bce,
            "total_loss": fit.total_loss,
            "r2": fit.r2,
            "pearson_r": fit.pearson_r,
        },
        "n_effect_pairs_available": int(len(available_pairs)),
        "effect_pairs": available_pairs,
        "unresolved_test_odors_encoded_as_zeros": sorted(set(unresolved_test)),
        "git_sha": _git_sha(),
    }
    save_json(out_dir / "metrics.json", metrics)

    run_config = {
        "behavior_csv": str(Path(args.behavior_csv).resolve()),
        "door_cache": str(Path(args.door_cache).resolve()),
        "output_dir": str(out_dir.resolve()),
        "train_prefix": args.train_prefix,
        "exclude_dataset": exclude_datasets,
        "seed": config.seed,
        "epochs": config.epochs,
        "lr": config.lr,
        "l1_lambda": config.l1_lambda,
        "include_test": config.include_test,
        "include_train": config.include_train,
        "include_interaction": config.include_interaction,
        "include_diff": config.include_diff,
        "include_reward": config.include_reward,
        "trained_odor_map_json": args.trained_odor_map_json,
        "connectome_dir": args.connectome_dir,
        "disable_connectome_analysis": bool(args.disable_connectome_analysis),
        "strict": bool(args.strict),
        "git_sha": metrics["git_sha"],
    }
    save_json(out_dir / "run_config.json", run_config)

    # Save training curve for debugging/diagnostics
    pd.DataFrame({"loss": loss_history}).to_csv(out_dir / "training_loss.csv", index=False)

    logger.info("Wrote: %s", out_dir / "predicted_rates.csv")
    logger.info("Wrote: %s", out_dir / "metrics.json")
    logger.info("Wrote: %s", out_dir / "orn_importance_global.csv")
    logger.info("Wrote: %s", out_dir / "ablation_sweep.csv")


if __name__ == "__main__":
    main()
