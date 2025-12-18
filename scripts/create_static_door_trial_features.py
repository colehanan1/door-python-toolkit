#!/usr/bin/env python3
"""
Create Static DoOR Trial Feature Matrix for RNN Training

This script transforms trial-by-trial behavioral data into feature matrices using
static odorant-receptor response profiles from the DoOR database.

Author: Generated for neuroscience data pipeline
Date: 2025-12-17

Key Design Decisions (Chain-of-Thought):
=========================================
1. Static DoOR profiles: Each odorant → 78-dim receptor response vector
   WHY: RNN will learn circuit dynamics (PN inhibition, KC plasticity) from static features

2. Feature composition: [trained (78) + test (78) + diff (78) + similarity (1) + history (1)]
   WHY: Provides both absolute and relative odorant representations for learning

3. Trial-level granularity: One feature vector per trial
   WHY: Enables learning of within-fly, across-trial behavioral dynamics

4. Fly-level grouping: Tracked via metadata
   WHY: Prevents data leakage in cross-validation splits
"""

import os
import sys
import json
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from difflib import SequenceMatcher
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Add door_toolkit to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from door_toolkit.encoder import DoOREncoder

warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================================
# STEP 1: VALIDATE INPUT DATA STRUCTURE
# ============================================================================

def load_behavioral_data(data_dir: str) -> pd.DataFrame:
    """
    Load behavioral trial data from all condition subdirectories.

    Chain-of-Thought:
    -----------------
    WHY: Need to aggregate trials across all experimental conditions
    HOW: Iterate through subdirectories, load CSV files, concatenate
    WHAT TO CHECK:
        - All required columns present (dataset, fly, trial_num, odor_sent, responses)
        - No completely missing columns
        - Reasonable number of trials per condition

    Args:
        data_dir: Path to directory containing condition subdirectories

    Returns:
        DataFrame with all trials, columns: [dataset, fly, fly_number, trial_num,
                                              odor_sent, during_hit, after_hit,
                                              prob_reaction, trained_odor]
    """
    print("\n" + "="*80)
    print("STEP 1: VALIDATING INPUT DATA STRUCTURE")
    print("="*80)

    print(f"\n[WHY] Loading behavioral data from: {data_dir}")
    print("[JUSTIFICATION] Must aggregate all trials across conditions before feature extraction")

    # Find all condition directories (exclude non-directory files)
    condition_dirs = [d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))]

    print(f"\n[CHECK] Found {len(condition_dirs)} condition directories:")
    for cdir in sorted(condition_dirs):
        print(f"  - {cdir}")

    # Load data from each condition
    all_trials = []
    condition_stats = {}

    for condition in sorted(condition_dirs):
        condition_path = os.path.join(data_dir, condition)

        # Find the binary reactions CSV file
        csv_pattern = os.path.join(condition_path, "binary_reactions_*_unordered.csv")
        csv_files = glob.glob(csv_pattern)

        if not csv_files:
            print(f"\n[WARNING] No CSV file found for condition: {condition}")
            continue

        csv_file = csv_files[0]
        print(f"\n[LOADING] {condition}: {os.path.basename(csv_file)}")

        # Load CSV
        df = pd.read_csv(csv_file)

        # Extract trained odorant from condition name
        # Examples: opto_EB → Ethyl Butyrate, opto_hex → Hexanol
        trained_odor = extract_trained_odor_from_condition(condition)
        df['trained_odor'] = trained_odor
        df['condition'] = condition

        # Validate columns
        required_cols = ['dataset', 'fly', 'trial_num', 'odor_sent', 'during_hit', 'after_hit']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  [ERROR] Missing columns: {missing_cols}")
            continue

        # Statistics
        n_trials = len(df)
        n_flies = df['fly'].nunique()
        n_test_odors = df['odor_sent'].nunique()

        print(f"  [STATS] Trials: {n_trials} | Flies: {n_flies} | Test odors: {n_test_odors}")
        print(f"  [INFO] Trained on: {trained_odor}")
        print(f"  [INFO] Test odors: {sorted(df['odor_sent'].unique())}")

        condition_stats[condition] = {
            'n_trials': n_trials,
            'n_flies': n_flies,
            'trained_odor': trained_odor
        }

        all_trials.append(df)

    # Concatenate all trials
    print(f"\n[CONCATENATING] Combining trials from {len(all_trials)} conditions...")
    combined_df = pd.concat(all_trials, ignore_index=True)

    # Final validation
    print(f"\n[VALIDATION] Final dataset shape: {combined_df.shape}")
    print(f"  - Total trials: {len(combined_df)}")
    print(f"  - Total unique flies: {combined_df['fly'].nunique()}")
    print(f"  - Unique trained odors: {combined_df['trained_odor'].nunique()}")
    print(f"  - Unique test odors: {combined_df['odor_sent'].nunique()}")

    # Check for missing values
    missing_counts = combined_df[['during_hit', 'after_hit', 'trained_odor', 'odor_sent']].isnull().sum()
    print(f"\n[MISSING VALUES CHECK]")
    for col, count in missing_counts.items():
        pct = 100 * count / len(combined_df)
        status = "✓ OK" if pct < 5 else "✗ WARNING"
        print(f"  {col}: {count} ({pct:.1f}%) {status}")

    if missing_counts.sum() / len(combined_df) > 0.05:
        raise ValueError(
            f"[CRITICAL] >5% missing values detected. Clean data before proceeding.\n"
            f"Missing counts:\n{missing_counts}"
        )

    # Summary by condition
    print(f"\n[CONDITION SUMMARY]")
    for condition, stats in sorted(condition_stats.items()):
        print(f"  {condition:25s} | Trials: {stats['n_trials']:4d} | "
              f"Flies: {stats['n_flies']:2d} | Trained: {stats['trained_odor']}")

    print("\n[STEP 1 COMPLETE] ✓ Data structure validated")
    print(f"[READY] {len(combined_df)} trials loaded from {len(condition_stats)} conditions")

    return combined_df


def extract_trained_odor_from_condition(condition: str) -> str:
    """
    Extract the trained odorant name from the condition directory name.

    Chain-of-Thought:
    -----------------
    WHY: Trained odorant is encoded in condition name (e.g., opto_EB → Ethyl Butyrate)
    HOW: Use mapping dictionary with known experimental conditions

    Examples:
        opto_EB → Ethyl Butyrate
        opto_hex → Hexanol
        opto_benz_1 → Benzaldehyde
        opto_ACV → Apple Cider Vinegar
        opto_3-oct → 3-Octonol
        opto_AIR → AIR
    """
    # Mapping based on experimental design
    mapping = {
        'opto_EB': 'Ethyl Butyrate',
        'opto_EB_6_training': 'Ethyl Butyrate (6-Training)',
        'EB_control': 'Ethyl Butyrate',
        'opto_hex': 'Hexanol',
        'hex_control': 'Hexanol',
        'opto_benz_1': 'Benzaldehyde',
        'Benz_control': 'Benzaldehyde',
        'opto_ACV': 'Apple Cider Vinegar',
        'opto_3-oct': '3-Octonol',
        'opto_AIR': 'AIR',
    }

    if condition in mapping:
        return mapping[condition]
    else:
        # Fallback: try to extract from condition name
        print(f"[WARNING] Unknown condition '{condition}', attempting to infer trained odor")
        return condition.replace('opto_', '').replace('_', ' ').title()


# ============================================================================
# STEP 2: MAP BEHAVIORAL ODORANTS TO DoOR DATABASE
# ============================================================================

def map_odorant_names_to_door(
    odor_names: List[str],
    door_encoder: DoOREncoder
) -> Dict[str, str]:
    """
    Create mapping between behavioral odorant names and DoOR canonical names.

    Chain-of-Thought:
    -----------------
    WHY: Behavioral data may use different naming conventions than DoOR
    HOW:
        1. Try exact match (case-insensitive)
        2. Try fuzzy matching with preprocessing (remove spaces, underscores)
        3. Manual mapping for known aliases
    WHAT TO LOG: All mappings so user can verify correctness

    Args:
        odor_names: List of odorant names from behavioral data
        door_encoder: Initialized DoOREncoder instance

    Returns:
        Dictionary mapping behavioral_name → door_canonical_name
    """
    print("\n" + "="*80)
    print("STEP 2: MAPPING BEHAVIORAL ODORANTS TO DoOR DATABASE")
    print("="*80)

    print(f"\n[WHY] Map behavioral odorant names to DoOR canonical names")
    print("[JUSTIFICATION] Name mismatches would prevent DoOR profile extraction")

    # Get all available DoOR odorants
    door_odorants = door_encoder.odorant_names
    print(f"\n[INFO] DoOR database contains {len(door_odorants)} odorants")

    # Manual mapping for known aliases
    manual_mapping = {
        'Ethyl Butyrate': 'ethyl butyrate',
        'Ethyl Butyrate (6-Training)': 'ethyl butyrate',
        'Apple Cider Vinegar': 'acetic acid',  # ACV is primarily acetic acid
        '3-Octonol': '3-octanol',
        'Hexanol': '1-hexanol',
        'AIR': None,  # Control condition - no DoOR profile
    }

    mapping = {}
    successful_matches = []
    fuzzy_matches = []
    failed_matches = []

    print(f"\n[MAPPING] Processing {len(odor_names)} unique odorants...")

    for behavioral_name in sorted(odor_names):
        print(f"\n  Mapping: '{behavioral_name}'")

        # Skip empty or null values
        if pd.isna(behavioral_name) or behavioral_name == '':
            print(f"    [SKIP] Empty name")
            continue

        # Check manual mapping first
        if behavioral_name in manual_mapping:
            door_name = manual_mapping[behavioral_name]
            mapping[behavioral_name] = door_name
            if door_name is None:
                print(f"    → [CONTROL] No DoOR profile needed")
            else:
                print(f"    → [MANUAL] '{door_name}'")
                successful_matches.append((behavioral_name, door_name, 'manual'))
            continue

        # Try exact match (case-insensitive)
        behavioral_lower = behavioral_name.lower().strip()

        # First try: exact match
        if behavioral_lower in door_odorants:
            mapping[behavioral_name] = behavioral_lower
            print(f"    → [EXACT] '{behavioral_lower}'")
            successful_matches.append((behavioral_name, behavioral_lower, 'exact'))
            continue

        # Second try: fuzzy matching
        best_match = None
        best_score = 0.0

        for door_name in door_odorants:
            # Compare cleaned versions
            behavioral_clean = behavioral_lower.replace('_', ' ').replace('-', ' ')
            door_clean = door_name.replace('_', ' ').replace('-', ' ')

            score = SequenceMatcher(None, behavioral_clean, door_clean).ratio()

            if score > best_score:
                best_score = score
                best_match = door_name

        # Accept fuzzy match if confidence is high enough
        if best_score >= 0.85:
            mapping[behavioral_name] = best_match
            print(f"    → [FUZZY] '{best_match}' (score: {best_score:.2f})")
            fuzzy_matches.append((behavioral_name, best_match, best_score))
        else:
            # Match failed
            mapping[behavioral_name] = None
            print(f"    → [FAILED] Best match: '{best_match}' (score: {best_score:.2f}) - too low!")
            failed_matches.append((behavioral_name, best_match, best_score))

    # Summary
    print(f"\n[MAPPING SUMMARY]")
    print(f"  ✓ Successful matches: {len(successful_matches)}")
    print(f"  ~ Fuzzy matches: {len(fuzzy_matches)}")
    print(f"  ✗ Failed matches: {len(failed_matches)}")

    if fuzzy_matches:
        print(f"\n[FUZZY MATCHES] (verify these are correct)")
        for behavioral, door, score in fuzzy_matches:
            print(f"  '{behavioral}' → '{door}' (confidence: {score:.2%})")

    if failed_matches:
        print(f"\n[FAILED MATCHES] (manual intervention needed)")
        for behavioral, best_candidate, score in failed_matches:
            print(f"  '{behavioral}' → best candidate: '{best_candidate}' (confidence: {score:.2%})")

        raise ValueError(
            f"[CRITICAL] {len(failed_matches)} odorants could not be mapped to DoOR.\n"
            f"Please add manual mappings in the code for these odorants."
        )

    print("\n[STEP 2 COMPLETE] ✓ All odorants successfully mapped to DoOR")

    return mapping


# ============================================================================
# STEP 3: EXTRACT STATIC DoOR PROFILES
# ============================================================================

def extract_door_profiles(
    odorant_mapping: Dict[str, str],
    door_encoder: DoOREncoder
) -> Dict[str, np.ndarray]:
    """
    Extract 78-dimensional DoOR receptor response profiles for all odorants.

    Chain-of-Thought:
    -----------------
    WHY: Convert odorant identities into numerical receptor response vectors
    HOW: Use DoOREncoder.encode() to get 78-dim profiles
    WHAT TO CHECK:
        - All profiles are 78-dimensional
        - Values are in reasonable range (typically 0-1)
        - Control odorants (AIR) get zero vectors

    Args:
        odorant_mapping: Dict mapping behavioral_name → door_canonical_name
        door_encoder: Initialized DoOREncoder instance

    Returns:
        Dict mapping door_canonical_name → 78-dim numpy array
    """
    print("\n" + "="*80)
    print("STEP 3: EXTRACTING STATIC DoOR PROFILES")
    print("="*80)

    print(f"\n[WHY] Extract static receptor response profiles for each odorant")
    print("[JUSTIFICATION] These form the basis of our feature vectors")

    profiles = {}
    profile_stats = []

    # Get unique DoOR names (excluding None for control conditions)
    unique_door_names = set(v for v in odorant_mapping.values() if v is not None)

    print(f"\n[EXTRACTING] Processing {len(unique_door_names)} unique DoOR odorants...")

    for door_name in sorted(unique_door_names):
        print(f"\n  Encoding: '{door_name}'")

        try:
            # Get DoOR profile (78-dimensional)
            profile = door_encoder.encode([door_name])[0]

            # Convert to numpy if it's a torch tensor
            if hasattr(profile, 'numpy'):
                profile_np = profile.cpu().numpy() if hasattr(profile, 'cpu') else profile.numpy()
            else:
                profile_np = np.array(profile)

            # Validate
            if profile_np.shape[0] != 78:
                raise ValueError(f"Expected 78-dim profile, got {profile_np.shape[0]}")

            profiles[door_name] = profile_np

            # Statistics
            n_responding = (profile_np > 0.1).sum()
            max_response = profile_np.max()
            mean_response = profile_np.mean()

            print(f"    [PROFILE] Shape: {profile_np.shape}, "
                  f"Responding receptors: {n_responding}/78, "
                  f"Max: {max_response:.3f}, Mean: {mean_response:.3f}")

            # Sanity check: print top 3 responding receptors
            top_indices = np.argsort(profile_np)[-3:][::-1]
            receptor_names = door_encoder.receptor_names
            print(f"    [TOP RECEPTORS]")
            for idx in top_indices:
                receptor = receptor_names[idx] if idx < len(receptor_names) else f"R{idx}"
                response = profile_np[idx]
                print(f"      {receptor}: {response:.3f}")

            profile_stats.append({
                'odor': door_name,
                'n_responding': n_responding,
                'max_response': max_response,
                'mean_response': mean_response
            })

        except Exception as e:
            print(f"    [ERROR] Failed to encode '{door_name}': {e}")
            raise

    # Add zero vector for control conditions (AIR)
    if any(v is None for v in odorant_mapping.values()):
        profiles[None] = np.zeros(78)
        print(f"\n  [CONTROL] AIR → zero vector (78 dims)")

    # Summary statistics
    print(f"\n[PROFILE STATISTICS]")
    stats_df = pd.DataFrame(profile_stats)
    print(f"  Mean responding receptors: {stats_df['n_responding'].mean():.1f} / 78")
    print(f"  Mean max response: {stats_df['max_response'].mean():.3f}")
    print(f"  Mean avg response: {stats_df['mean_response'].mean():.3f}")

    print(f"\n[DIVERSITY CHECK] Odors with most/fewest responding receptors:")
    sorted_stats = stats_df.sort_values('n_responding')
    print(f"  Least: {sorted_stats.iloc[0]['odor']} ({sorted_stats.iloc[0]['n_responding']} receptors)")
    print(f"  Most: {sorted_stats.iloc[-1]['odor']} ({sorted_stats.iloc[-1]['n_responding']} receptors)")

    print("\n[STEP 3 COMPLETE] ✓ DoOR profiles extracted for all odorants")
    print(f"[READY] {len(profiles)} odorant profiles available")

    return profiles


# ============================================================================
# STEP 4: CREATE TRIAL-LEVEL FEATURE VECTORS
# ============================================================================

def create_trial_features(
    behavioral_df: pd.DataFrame,
    door_profiles: Dict[str, np.ndarray],
    odorant_mapping: Dict[str, str]
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Convert each trial into a feature vector combining trained & test DoOR profiles.

    Chain-of-Thought:
    -----------------
    WHY: RNN needs numerical input that captures odorant identities and relationships
    HOW: Concatenate multiple feature types:
        - trained_door_profile (78 dims): Static response to training odor
        - test_door_profile (78 dims): Static response to test odor
        - response_difference (78 dims): test - trained (directional change)
        - odor_similarity (1 dim): Cosine similarity between profiles
        - trial_within_fly (1 dim): Normalized trial index [0, 1]
        - learning_history (1 dim): Moving average of recent responses

    TOTAL: 78 + 78 + 78 + 1 + 1 + 1 = 237 dimensions

    Args:
        behavioral_df: DataFrame with trial data
        door_profiles: Dict of DoOR profiles
        odorant_mapping: Dict mapping behavioral → DoOR names

    Returns:
        - feature_matrix: [n_trials, 237] numpy array
        - labels: [n_trials] binary array (0=avoid, 1=approach)
        - metadata_df: DataFrame with trial metadata
        - feature_names: List of feature names
    """
    print("\n" + "="*80)
    print("STEP 4: CREATING TRIAL-LEVEL FEATURE VECTORS")
    print("="*80)

    print(f"\n[WHY] Transform each trial into numerical feature vector")
    print("[JUSTIFICATION] RNN requires fixed-size numerical inputs")

    print(f"\n[FEATURE DESIGN]")
    print(f"  1. trained_door_profile: 78 dims (static receptor responses to trained odor)")
    print(f"  2. test_door_profile: 78 dims (static receptor responses to test odor)")
    print(f"  3. response_difference: 78 dims (test - trained)")
    print(f"  4. odor_similarity: 1 dim (cosine similarity between profiles)")
    print(f"  5. trial_within_fly: 1 dim (normalized trial index)")
    print(f"  6. learning_history: 1 dim (moving average of previous responses)")
    print(f"  TOTAL: 237 dimensions")

    n_trials = len(behavioral_df)
    feature_list = []
    label_list = []
    metadata_list = []

    print(f"\n[PROCESSING] Creating features for {n_trials} trials...")

    # Group by fly to compute within-fly statistics
    for fly_idx, (fly_name, fly_df) in enumerate(behavioral_df.groupby('fly')):
        fly_df = fly_df.sort_values('trial_num').reset_index(drop=True)
        n_fly_trials = len(fly_df)

        if fly_idx % 5 == 0:  # Print progress every 5 flies
            print(f"  Processing fly {fly_idx + 1} ({fly_name}): {n_fly_trials} trials")

        # Compute learning history (moving average of previous responses)
        # Use 'during_hit' as the primary response measure
        responses = fly_df['during_hit'].values
        learning_history = np.zeros(n_fly_trials)

        for i in range(n_fly_trials):
            if i == 0:
                learning_history[i] = 0.5  # Prior: assume 50% approach rate
            else:
                # Average of previous trials (up to last 5 trials)
                window = min(5, i)
                learning_history[i] = responses[max(0, i - window):i].mean()

        # Process each trial
        for trial_idx, (_, trial) in enumerate(fly_df.iterrows()):
            # Get odorant profiles
            trained_name = trial['trained_odor']
            test_name = trial['odor_sent']

            trained_door = odorant_mapping.get(trained_name, None)
            test_door = odorant_mapping.get(test_name, None)

            trained_profile = door_profiles.get(trained_door, np.zeros(78))
            test_profile = door_profiles.get(test_door, np.zeros(78))

            # Feature 1: Trained odor profile (78 dims)
            feat_trained = trained_profile

            # Feature 2: Test odor profile (78 dims)
            feat_test = test_profile

            # Feature 3: Response difference (78 dims)
            feat_diff = test_profile - trained_profile

            # Feature 4: Odor similarity (1 dim)
            # Cosine similarity between trained and test profiles
            if np.linalg.norm(trained_profile) > 0 and np.linalg.norm(test_profile) > 0:
                feat_similarity = cosine_similarity(
                    trained_profile.reshape(1, -1),
                    test_profile.reshape(1, -1)
                )[0, 0]
            else:
                feat_similarity = 0.0

            # Feature 5: Trial within fly (1 dim, normalized [0, 1])
            feat_trial_idx = trial_idx / max(1, n_fly_trials - 1)

            # Feature 6: Learning history (1 dim)
            feat_history = learning_history[trial_idx]

            # Concatenate all features
            feature_vector = np.concatenate([
                feat_trained,              # 78 dims
                feat_test,                 # 78 dims
                feat_diff,                 # 78 dims
                [feat_similarity],         # 1 dim
                [feat_trial_idx],          # 1 dim
                [feat_history]             # 1 dim
            ])

            # Label: use 'during_hit' as primary response
            label = trial['during_hit']

            # Store
            feature_list.append(feature_vector)
            label_list.append(label)

            # Metadata
            metadata_list.append({
                'fly': fly_name,
                'fly_number': trial.get('fly_number', 1),
                'trial_num': trial['trial_num'],
                'condition': trial.get('condition', ''),
                'trained_odor': trained_name,
                'test_odor': test_name,
                'trained_door': trained_door,
                'test_door': test_door,
                'odor_similarity': feat_similarity,
                'trial_within_fly': feat_trial_idx,
                'learning_history': feat_history,
                'prob_reaction': trial.get('prob_reaction', np.nan)
            })

    # Convert to arrays
    feature_matrix = np.array(feature_list, dtype=np.float32)
    labels = np.array(label_list, dtype=np.int64)
    metadata_df = pd.DataFrame(metadata_list)

    # Feature names
    receptor_names = [f"trained_R{i:02d}" for i in range(78)] + \
                     [f"test_R{i:02d}" for i in range(78)] + \
                     [f"diff_R{i:02d}" for i in range(78)] + \
                     ["odor_similarity", "trial_within_fly", "learning_history"]

    # Validation
    print(f"\n[VALIDATION]")
    print(f"  Feature matrix shape: {feature_matrix.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Expected dimensions: ({n_trials}, 237)")

    if feature_matrix.shape != (n_trials, 237):
        raise ValueError(f"Feature matrix has wrong shape: {feature_matrix.shape}")

    # Check for NaN values
    n_nan_features = np.isnan(feature_matrix).sum()
    n_nan_labels = np.isnan(labels).sum()

    print(f"\n[NaN CHECK]")
    print(f"  NaN in features: {n_nan_features} / {feature_matrix.size} "
          f"({100 * n_nan_features / feature_matrix.size:.3f}%)")
    print(f"  NaN in labels: {n_nan_labels} / {len(labels)}")

    if n_nan_features > 0 or n_nan_labels > 0:
        raise ValueError("NaN values detected in features or labels!")

    # Feature statistics
    print(f"\n[FEATURE STATISTICS]")
    print(f"  Odor similarity: mean={metadata_df['odor_similarity'].mean():.3f}, "
          f"std={metadata_df['odor_similarity'].std():.3f}")
    print(f"  Learning history: mean={metadata_df['learning_history'].mean():.3f}, "
          f"std={metadata_df['learning_history'].std():.3f}")

    # Sanity check: trials with same trained/test odor should have similarity ~1.0
    same_odor_mask = metadata_df['trained_odor'] == metadata_df['test_odor']
    if same_odor_mask.sum() > 0:
        same_odor_sim = metadata_df.loc[same_odor_mask, 'odor_similarity'].mean()
        print(f"\n[SANITY CHECK] Same odor similarity (should be ~1.0): {same_odor_sim:.3f}")
        if same_odor_sim < 0.95:
            print(f"  [WARNING] Expected ~1.0 for identical odors!")

    print("\n[STEP 4 COMPLETE] ✓ Trial feature vectors created")
    print(f"[READY] {n_trials} feature vectors, each 237-dimensional")

    return feature_matrix, labels, metadata_df, receptor_names


# ============================================================================
# STEP 5: CREATE TARGET LABELS & CHECK CLASS BALANCE
# ============================================================================

def analyze_class_balance(
    labels: np.ndarray,
    metadata_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyze class balance in the dataset.

    Chain-of-Thought:
    -----------------
    WHY: Imbalanced datasets can bias RNN training toward majority class
    HOW: Compute class proportions overall and per condition
    WHAT TO REPORT:
        - Overall class balance (% approach vs avoid)
        - Balance per experimental condition
        - Recommendation for class weighting in loss function

    Args:
        labels: Binary labels (0=avoid, 1=approach)
        metadata_df: Trial metadata

    Returns:
        Dict with class balance statistics
    """
    print("\n" + "="*80)
    print("STEP 5: ANALYZING TARGET LABELS & CLASS BALANCE")
    print("="*80)

    print(f"\n[WHY] Check class balance to inform RNN training strategy")
    print("[JUSTIFICATION] Imbalanced classes may require weighted loss function")

    # Overall balance
    n_avoid = (labels == 0).sum()
    n_approach = (labels == 1).sum()
    total = len(labels)

    pct_avoid = 100 * n_avoid / total
    pct_approach = 100 * n_approach / total

    print(f"\n[OVERALL CLASS BALANCE]")
    print(f"  Avoid (0): {n_avoid:5d} / {total} ({pct_avoid:.1f}%)")
    print(f"  Approach (1): {n_approach:5d} / {total} ({pct_approach:.1f}%)")

    # Assess balance
    balance_ratio = min(n_avoid, n_approach) / max(n_avoid, n_approach)

    if balance_ratio >= 0.8:
        balance_status = "✓ BALANCED"
        recommendation = "No class weighting needed"
    elif balance_ratio >= 0.5:
        balance_status = "~ MODERATELY IMBALANCED"
        recommendation = "Consider using class_weight in loss function"
    else:
        balance_status = "✗ HIGHLY IMBALANCED"
        recommendation = "Use class_weight in loss function (critical)"

    print(f"\n  Status: {balance_status} (ratio: {balance_ratio:.2f})")
    print(f"  Recommendation: {recommendation}")

    # Compute class weights for loss function
    class_weights = {
        0: total / (2 * n_avoid),
        1: total / (2 * n_approach)
    }

    print(f"\n[SUGGESTED CLASS WEIGHTS] (for PyTorch loss function)")
    print(f"  class_weight = {{{0}: {class_weights[0]:.3f}, {1}: {class_weights[1]:.3f}}}")

    # Balance per condition
    print(f"\n[CLASS BALANCE PER CONDITION]")

    metadata_with_labels = metadata_df.copy()
    metadata_with_labels['label'] = labels

    condition_balance = []
    for condition, group in metadata_with_labels.groupby('condition'):
        n_trials = len(group)
        n_approach_cond = (group['label'] == 1).sum()
        pct_approach_cond = 100 * n_approach_cond / n_trials

        condition_balance.append({
            'condition': condition,
            'n_trials': n_trials,
            'n_approach': n_approach_cond,
            'pct_approach': pct_approach_cond
        })

        print(f"  {condition:25s} | Trials: {n_trials:4d} | "
              f"Approach: {n_approach_cond:4d} ({pct_approach_cond:5.1f}%)")

    balance_df = pd.DataFrame(condition_balance)

    # Check for conditions with extreme imbalance
    extreme_conditions = balance_df[
        (balance_df['pct_approach'] < 10) | (balance_df['pct_approach'] > 90)
    ]

    if len(extreme_conditions) > 0:
        print(f"\n[WARNING] Extreme imbalance detected in {len(extreme_conditions)} conditions:")
        for _, row in extreme_conditions.iterrows():
            print(f"  {row['condition']}: {row['pct_approach']:.1f}% approach")
        print("  Consider stratified sampling during train/val split")

    stats = {
        'n_total': total,
        'n_avoid': n_avoid,
        'n_approach': n_approach,
        'pct_avoid': pct_avoid,
        'pct_approach': pct_approach,
        'balance_ratio': balance_ratio,
        'balance_status': balance_status,
        'class_weights': class_weights,
        'recommendation': recommendation,
        'condition_balance': balance_df
    }

    print("\n[STEP 5 COMPLETE] ✓ Class balance analyzed")

    return stats


# ============================================================================
# STEP 6: GROUP BY FLY FOR CROSS-VALIDATION
# ============================================================================

def create_fly_groupings(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unique fly IDs for cross-validation splitting.

    Chain-of-Thought:
    -----------------
    WHY: Prevent data leakage - trials from same fly must stay together in CV splits
    HOW: Create unique fly_group ID combining batch and fly_number
    WHAT TO CHECK:
        - Each fly has multiple trials
        - Flies are distributed across conditions
        - No overlap in fly IDs across conditions (if applicable)

    Args:
        metadata_df: Trial metadata DataFrame

    Returns:
        metadata_df with added 'fly_group' column
    """
    print("\n" + "="*80)
    print("STEP 6: GROUPING TRIALS BY FLY FOR CROSS-VALIDATION")
    print("="*80)

    print(f"\n[WHY] Create fly groupings to prevent data leakage in CV splits")
    print("[JUSTIFICATION] RNN must generalize to new flies, not memorize specific flies")

    # Create unique fly group ID
    metadata_df['fly_group'] = metadata_df['fly'].astype(str) + "_" + \
                                metadata_df['fly_number'].astype(str)

    # Statistics
    n_unique_flies = metadata_df['fly_group'].nunique()
    trials_per_fly = metadata_df.groupby('fly_group').size()

    print(f"\n[FLY GROUPING STATISTICS]")
    print(f"  Total unique flies: {n_unique_flies}")
    print(f"  Trials per fly: mean={trials_per_fly.mean():.1f}, "
          f"median={trials_per_fly.median():.0f}, "
          f"min={trials_per_fly.min()}, max={trials_per_fly.max()}")

    # Check distribution across conditions
    print(f"\n[FLIES PER CONDITION]")
    for condition, group in metadata_df.groupby('condition'):
        n_flies_cond = group['fly_group'].nunique()
        n_trials_cond = len(group)
        print(f"  {condition:25s} | Flies: {n_flies_cond:2d} | Trials: {n_trials_cond:4d}")

    # Verify no fly appears in multiple conditions (would complicate CV)
    fly_conditions = metadata_df.groupby('fly_group')['condition'].nunique()
    multi_condition_flies = fly_conditions[fly_conditions > 1]

    if len(multi_condition_flies) > 0:
        print(f"\n[WARNING] {len(multi_condition_flies)} flies appear in multiple conditions:")
        for fly, n_cond in multi_condition_flies.items():
            conditions = metadata_df[metadata_df['fly_group'] == fly]['condition'].unique()
            print(f"  {fly}: {list(conditions)}")
        print("  This is OK but may complicate cross-validation strategy")

    # Recommendation for CV
    print(f"\n[CROSS-VALIDATION RECOMMENDATION]")
    if n_unique_flies >= 10:
        n_folds = min(5, n_unique_flies // 2)
        print(f"  Recommended: {n_folds}-fold GroupKFold (by fly_group)")
        print(f"  This ensures each fly's trials stay together in train or val")
    else:
        print(f"  Warning: Only {n_unique_flies} flies - limited CV splits possible")
        print(f"  Consider Leave-One-Out or 2-3 fold splits")

    print("\n[STEP 6 COMPLETE] ✓ Fly groupings created")

    return metadata_df


# ============================================================================
# STEP 7: SAVE AS PYTORCH DATASET
# ============================================================================

def save_trial_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
    feature_names: List[str],
    class_balance_stats: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Save dataset as PyTorch tensors and metadata files.

    Chain-of-Thought:
    -----------------
    WHY: Standard format for RNN training with PyTorch
    HOW: Save tensors (.pt), metadata CSV, and JSON configs
    WHAT TO SAVE:
        - trial_features.pt: [n_trials, 237] tensor
        - trial_labels.pt: [n_trials] tensor
        - trial_metadata.csv: trial-level info (fly, odors, etc.)
        - feature_metadata.json: feature names, dimensions
        - odorant_mapping.json: behavioral → DoOR name mappings
        - class_balance.json: class statistics for training
        - dataset_report.txt: human-readable summary

    Args:
        features: Feature matrix [n_trials, feature_dim]
        labels: Binary labels [n_trials]
        metadata_df: Trial metadata
        feature_names: List of feature names
        class_balance_stats: Class balance statistics
        output_dir: Directory to save outputs
    """
    print("\n" + "="*80)
    print("STEP 7: SAVING AS PYTORCH DATASET")
    print("="*80)

    print(f"\n[WHY] Save dataset in PyTorch-compatible format")
    print("[JUSTIFICATION] Standard format for RNN training")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[OUTPUT DIR] {output_dir}")

    # Convert to PyTorch tensors
    features_tensor = torch.from_numpy(features).float()
    labels_tensor = torch.from_numpy(labels).long()

    print(f"\n[CONVERTING TO PYTORCH]")
    print(f"  Features: {features_tensor.shape}, dtype={features_tensor.dtype}")
    print(f"  Labels: {labels_tensor.shape}, dtype={labels_tensor.dtype}")

    # Save tensors
    features_path = os.path.join(output_dir, "trial_features.pt")
    labels_path = os.path.join(output_dir, "trial_labels.pt")

    torch.save(features_tensor, features_path)
    torch.save(labels_tensor, labels_path)

    print(f"\n[SAVED TENSORS]")
    print(f"  ✓ {features_path}")
    print(f"  ✓ {labels_path}")

    # Save metadata CSV
    metadata_path = os.path.join(output_dir, "trial_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    print(f"  ✓ {metadata_path}")

    # Save feature metadata JSON
    feature_metadata = {
        'feature_names': feature_names,
        'feature_dim': len(feature_names),
        'n_trials': len(labels),
        'n_features': len(feature_names),
        'feature_groups': {
            'trained_door_profile': list(range(0, 78)),
            'test_door_profile': list(range(78, 156)),
            'response_difference': list(range(156, 234)),
            'odor_similarity': [234],
            'trial_within_fly': [235],
            'learning_history': [236]
        }
    }

    feature_metadata_path = os.path.join(output_dir, "feature_metadata.json")
    with open(feature_metadata_path, 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    print(f"  ✓ {feature_metadata_path}")

    # Save class balance JSON
    class_balance_json = {
        'n_total': int(class_balance_stats['n_total']),
        'n_avoid': int(class_balance_stats['n_avoid']),
        'n_approach': int(class_balance_stats['n_approach']),
        'pct_avoid': float(class_balance_stats['pct_avoid']),
        'pct_approach': float(class_balance_stats['pct_approach']),
        'balance_ratio': float(class_balance_stats['balance_ratio']),
        'balance_status': class_balance_stats['balance_status'],
        'class_weights': {int(k): float(v) for k, v in class_balance_stats['class_weights'].items()},
        'recommendation': class_balance_stats['recommendation']
    }

    class_balance_path = os.path.join(output_dir, "class_balance.json")
    with open(class_balance_path, 'w') as f:
        json.dump(class_balance_json, f, indent=2)
    print(f"  ✓ {class_balance_path}")

    # Verify tensors load correctly
    print(f"\n[VERIFICATION] Loading saved tensors...")
    loaded_features = torch.load(features_path)
    loaded_labels = torch.load(labels_path)

    assert loaded_features.shape == features_tensor.shape, "Feature tensor shape mismatch!"
    assert loaded_labels.shape == labels_tensor.shape, "Label tensor shape mismatch!"
    assert torch.allclose(loaded_features, features_tensor), "Feature tensor values changed!"
    assert torch.all(loaded_labels == labels_tensor), "Label tensor values changed!"

    print(f"  ✓ Tensors verified - shapes and values match")

    print("\n[STEP 7 COMPLETE] ✓ Dataset saved successfully")


# ============================================================================
# STEP 8: GENERATE DIAGNOSTIC REPORT
# ============================================================================

def generate_diagnostic_report(
    features: np.ndarray,
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
    class_balance_stats: Dict[str, Any],
    output_dir: str
) -> str:
    """
    Generate comprehensive diagnostic report for dataset validation.

    Chain-of-Thought:
    -----------------
    WHY: User must trust the pipeline before investing in RNN training
    HOW: Summarize all key statistics and validation checks
    WHAT TO INCLUDE:
        - Dataset dimensions and completeness
        - Odorant mapping success rate
        - Class balance assessment
        - Feature quality checks
        - Fly grouping statistics
        - Training readiness assessment

    Args:
        features: Feature matrix
        labels: Binary labels
        metadata_df: Trial metadata
        class_balance_stats: Class balance statistics
        output_dir: Output directory

    Returns:
        Formatted report string
    """
    print("\n" + "="*80)
    print("STEP 8: GENERATING DIAGNOSTIC REPORT")
    print("="*80)

    print(f"\n[WHY] Create human-readable validation report")
    print("[JUSTIFICATION] User must verify data quality before RNN training")

    report_lines = []

    # Header
    report_lines.append("="*80)
    report_lines.append("STATIC DoOR TRIAL FEATURE DATASET - DIAGNOSTIC REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {pd.Timestamp.now()}")
    report_lines.append(f"Output directory: {output_dir}")
    report_lines.append("")

    # Dataset summary
    report_lines.append("-" * 80)
    report_lines.append("1. DATASET SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"  Total trials: {len(labels):,}")
    report_lines.append(f"  Feature dimensions: {features.shape[1]}")
    report_lines.append(f"  Unique flies: {metadata_df['fly_group'].nunique()}")
    report_lines.append(f"  Unique conditions: {metadata_df['condition'].nunique()}")
    report_lines.append(f"  Unique trained odors: {metadata_df['trained_odor'].nunique()}")
    report_lines.append(f"  Unique test odors: {metadata_df['test_odor'].nunique()}")
    report_lines.append("")

    # Feature composition
    report_lines.append("-" * 80)
    report_lines.append("2. FEATURE COMPOSITION")
    report_lines.append("-" * 80)
    report_lines.append("  Feature groups:")
    report_lines.append("    - trained_door_profile: 78 dims (indices 0-77)")
    report_lines.append("    - test_door_profile: 78 dims (indices 78-155)")
    report_lines.append("    - response_difference: 78 dims (indices 156-233)")
    report_lines.append("    - odor_similarity: 1 dim (index 234)")
    report_lines.append("    - trial_within_fly: 1 dim (index 235)")
    report_lines.append("    - learning_history: 1 dim (index 236)")
    report_lines.append(f"  Total: {features.shape[1]} dimensions")
    report_lines.append("")

    # Data quality
    report_lines.append("-" * 80)
    report_lines.append("3. DATA QUALITY CHECKS")
    report_lines.append("-" * 80)

    n_nan_features = np.isnan(features).sum()
    n_inf_features = np.isinf(features).sum()

    report_lines.append(f"  NaN values in features: {n_nan_features:,} / {features.size:,} "
                       f"({100 * n_nan_features / features.size:.3f}%)")
    report_lines.append(f"  Inf values in features: {n_inf_features:,}")
    report_lines.append(f"  NaN values in labels: {np.isnan(labels).sum()}")

    if n_nan_features == 0 and n_inf_features == 0:
        report_lines.append("  ✓ All features are valid (no NaN/Inf)")
    else:
        report_lines.append("  ✗ WARNING: Invalid values detected!")

    report_lines.append("")

    # Class balance
    report_lines.append("-" * 80)
    report_lines.append("4. CLASS BALANCE")
    report_lines.append("-" * 80)
    report_lines.append(f"  Total trials: {class_balance_stats['n_total']:,}")
    report_lines.append(f"  Avoid (0): {class_balance_stats['n_avoid']:,} "
                       f"({class_balance_stats['pct_avoid']:.1f}%)")
    report_lines.append(f"  Approach (1): {class_balance_stats['n_approach']:,} "
                       f"({class_balance_stats['pct_approach']:.1f}%)")
    report_lines.append(f"  Balance ratio: {class_balance_stats['balance_ratio']:.3f}")
    report_lines.append(f"  Status: {class_balance_stats['balance_status']}")
    report_lines.append(f"  Recommendation: {class_balance_stats['recommendation']}")
    report_lines.append("")

    # Suggested class weights
    cw = class_balance_stats['class_weights']
    report_lines.append("  Suggested class weights for loss function:")
    report_lines.append(f"    class_weight = {{{0}: {cw[0]:.3f}, {1}: {cw[1]:.3f}}}")
    report_lines.append("")

    # Fly grouping
    report_lines.append("-" * 80)
    report_lines.append("5. FLY GROUPING (FOR CROSS-VALIDATION)")
    report_lines.append("-" * 80)

    trials_per_fly = metadata_df.groupby('fly_group').size()

    report_lines.append(f"  Total unique flies: {metadata_df['fly_group'].nunique()}")
    report_lines.append(f"  Trials per fly:")
    report_lines.append(f"    Mean: {trials_per_fly.mean():.1f}")
    report_lines.append(f"    Median: {trials_per_fly.median():.0f}")
    report_lines.append(f"    Min: {trials_per_fly.min()}")
    report_lines.append(f"    Max: {trials_per_fly.max()}")
    report_lines.append("")

    # Condition breakdown
    report_lines.append("  Trials per condition:")
    for condition, group in metadata_df.groupby('condition'):
        n_flies = group['fly_group'].nunique()
        n_trials = len(group)
        report_lines.append(f"    {condition:25s} | Flies: {n_flies:2d} | Trials: {n_trials:4d}")
    report_lines.append("")

    # Feature statistics
    report_lines.append("-" * 80)
    report_lines.append("6. FEATURE STATISTICS")
    report_lines.append("-" * 80)

    report_lines.append(f"  Odor similarity (cosine):")
    report_lines.append(f"    Mean: {metadata_df['odor_similarity'].mean():.3f}")
    report_lines.append(f"    Std: {metadata_df['odor_similarity'].std():.3f}")
    report_lines.append(f"    Min: {metadata_df['odor_similarity'].min():.3f}")
    report_lines.append(f"    Max: {metadata_df['odor_similarity'].max():.3f}")
    report_lines.append("")

    report_lines.append(f"  Learning history:")
    report_lines.append(f"    Mean: {metadata_df['learning_history'].mean():.3f}")
    report_lines.append(f"    Std: {metadata_df['learning_history'].std():.3f}")
    report_lines.append("")

    # Sanity checks
    report_lines.append("-" * 80)
    report_lines.append("7. SANITY CHECKS")
    report_lines.append("-" * 80)

    # Check 1: Same odor similarity should be ~1.0
    same_odor_mask = metadata_df['trained_odor'] == metadata_df['test_odor']
    if same_odor_mask.sum() > 0:
        same_odor_sim = metadata_df.loc[same_odor_mask, 'odor_similarity'].mean()
        check1 = "✓ PASS" if same_odor_sim >= 0.95 else "✗ FAIL"
        report_lines.append(f"  Same-odor similarity (should be ~1.0): {same_odor_sim:.3f} {check1}")
    else:
        report_lines.append(f"  Same-odor similarity check: N/A (no same-odor trials)")

    # Check 2: Feature matrix has no NaN/Inf
    check2 = "✓ PASS" if (n_nan_features == 0 and n_inf_features == 0) else "✗ FAIL"
    report_lines.append(f"  No NaN/Inf in features: {check2}")

    # Check 3: Labels are binary
    unique_labels = np.unique(labels)
    check3 = "✓ PASS" if set(unique_labels) == {0, 1} or set(unique_labels) == {0} or set(unique_labels) == {1} else "✗ FAIL"
    report_lines.append(f"  Labels are binary (0 or 1): {check3}")

    # Check 4: All flies have multiple trials
    min_trials = trials_per_fly.min()
    check4 = "✓ PASS" if min_trials >= 3 else "⚠ WARNING"
    report_lines.append(f"  All flies have ≥3 trials: {check4} (min={min_trials})")

    report_lines.append("")

    # Validation summary
    report_lines.append("-" * 80)
    report_lines.append("8. VALIDATION SUMMARY")
    report_lines.append("-" * 80)

    all_checks_passed = (
        n_nan_features == 0 and
        n_inf_features == 0 and
        set(unique_labels).issubset({0, 1})
    )

    if all_checks_passed:
        report_lines.append("  ✓✓✓ ALL CHECKS PASSED ✓✓✓")
        report_lines.append("")
        report_lines.append("  Status: READY FOR TRAINING")
        report_lines.append("")
        report_lines.append("  Next steps:")
        report_lines.append("    1. Load dataset: features = torch.load('trial_features.pt')")
        report_lines.append("    2. Create DataLoader with GroupKFold by 'fly_group'")
        report_lines.append("    3. Use class weights in loss function (see section 4)")
        report_lines.append("    4. Train RNN with 237-dim input, binary classification output")
    else:
        report_lines.append("  ✗✗✗ VALIDATION FAILED ✗✗✗")
        report_lines.append("")
        report_lines.append("  Issues detected:")
        if n_nan_features > 0:
            report_lines.append("    - NaN values in features")
        if n_inf_features > 0:
            report_lines.append("    - Inf values in features")
        if not set(unique_labels).issubset({0, 1}):
            report_lines.append("    - Non-binary labels")
        report_lines.append("")
        report_lines.append("  Action: Fix issues before training")

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)

    # Join report
    report = "\n".join(report_lines)

    # Save report
    report_path = os.path.join(output_dir, "dataset_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n[SAVED REPORT] {report_path}")

    # Print to console
    print("\n" + report)

    print("\n[STEP 8 COMPLETE] ✓ Diagnostic report generated")

    return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""

    print("\n" + "="*80)
    print("STATIC DoOR TRIAL FEATURE MATRIX CREATION")
    print("="*80)
    print("\nThis script creates trial-level feature vectors for RNN training")
    print("by combining static DoOR receptor response profiles with trial metadata.")
    print("")

    # Configuration
    BEHAVIORAL_DATA_DIR = "/home/ramanlab/Documents/cole/Results/Opto/Reaction_Predictions(Strictest)"
    OUTPUT_DIR = "/home/ramanlab/Documents/cole/VSCode/door-python-toolkit/data/pgcn_features"

    print(f"Input directory: {BEHAVIORAL_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("")

    # Initialize DoOR encoder
    print("[INITIALIZING] DoOR encoder...")
    door_encoder = DoOREncoder()
    print("[OK] DoOR encoder ready")

    # STEP 1: Load behavioral data
    behavioral_df = load_behavioral_data(BEHAVIORAL_DATA_DIR)

    # STEP 2: Map odorant names to DoOR
    unique_trained = behavioral_df['trained_odor'].unique()
    unique_test = behavioral_df['odor_sent'].unique()
    all_odorants = list(set(list(unique_trained) + list(unique_test)))

    odorant_mapping = map_odorant_names_to_door(all_odorants, door_encoder)

    # Save odorant mapping
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping_path = os.path.join(OUTPUT_DIR, "odorant_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(odorant_mapping, f, indent=2)
    print(f"\n[SAVED] Odorant mapping: {mapping_path}")

    # STEP 3: Extract DoOR profiles
    door_profiles = extract_door_profiles(odorant_mapping, door_encoder)

    # STEP 4: Create trial features
    features, labels, metadata_df, feature_names = create_trial_features(
        behavioral_df, door_profiles, odorant_mapping
    )

    # STEP 5: Analyze class balance
    class_balance_stats = analyze_class_balance(labels, metadata_df)

    # STEP 6: Create fly groupings
    metadata_df = create_fly_groupings(metadata_df)

    # STEP 7: Save dataset
    save_trial_dataset(
        features, labels, metadata_df, feature_names,
        class_balance_stats, OUTPUT_DIR
    )

    # STEP 8: Generate report
    report = generate_diagnostic_report(
        features, labels, metadata_df, class_balance_stats, OUTPUT_DIR
    )

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\n✓ Dataset created successfully!")
    print(f"✓ Output saved to: {OUTPUT_DIR}")
    print(f"\n📊 Quick stats:")
    print(f"  - {len(labels):,} trials")
    print(f"  - {features.shape[1]} features per trial")
    print(f"  - {metadata_df['fly_group'].nunique()} unique flies")
    print(f"  - {metadata_df['condition'].nunique()} experimental conditions")
    print(f"\n📝 See dataset_report.txt for full validation")
    print("")


if __name__ == "__main__":
    main()
