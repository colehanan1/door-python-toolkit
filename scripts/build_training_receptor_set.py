#!/usr/bin/env python3
"""
Build Training Receptor Set - Single Source of Truth

This script creates the authoritative training receptor set by filtering
the receptor inventory for:
- Adult life stage only
- Mapped to valid ORN_* FlyWire targets
- Non-ambiguous mappings (unless --include_ambiguous flag is set)

Outputs:
- data/mappings/training_receptor_set.json
- data/mappings/training_receptor_set.csv

The training set includes provenance (hashes, timestamp) for full auditability.

Decision → Evidence → Implementation
-----------------------------------
Decision: Define training set as adult + mapped + non-ambiguous by default.
Evidence: Larval receptors target different circuits (Berck et al., eLife 2016).
         Ambiguous mappings introduce uncertainty in connectome constraints.
Implementation: Filter inventory by life_stage==Adult, is_mapped==Yes,
                is_ambiguous==No (unless flag overrides).
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


INVENTORY_FILE = Path("data/mappings/receptor_inventory.csv")
MAPPING_FILE = Path("data/mappings/door_to_flywire_mapping.csv")
OUTPUT_JSON = Path("data/mappings/training_receptor_set.json")
OUTPUT_CSV = Path("data/mappings/training_receptor_set.csv")


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of file for provenance."""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_inventory(inventory_path: Path) -> pd.DataFrame:
    """Load receptor inventory CSV."""
    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Receptor inventory not found: {inventory_path}\n"
            f"Run: python scripts/generate_receptor_inventory.py"
        )
    return pd.read_csv(inventory_path)


def load_mapping(mapping_path: Path) -> pd.DataFrame:
    """Load DoOR→FlyWire mapping CSV."""
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {mapping_path}\n"
            f"Run: python scripts/generate_complete_receptor_mapping.py"
        )
    return pd.read_csv(mapping_path)


def filter_training_receptors(
    inventory_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    include_ambiguous: bool = False,
    strict_adult_only: bool = True,
) -> pd.DataFrame:
    """
    Filter inventory for training-eligible receptors.

    Criteria:
    1. life_stage == "Adult" (if strict_adult_only=True)
    2. is_mapped == "Yes"
    3. is_ambiguous == "No" (unless include_ambiguous=True) [from mapping]
    4. flywire_target starts with "ORN_"
    """
    df = inventory_df.copy()

    # Rename receptor_name to door_name for consistency
    if 'receptor_name' in df.columns:
        df = df.rename(columns={'receptor_name': 'door_name'})

    # Filter 1: Adult only
    if strict_adult_only:
        adult_mask = df['life_stage'].astype(str).str.strip() == 'Adult'
        df = df[adult_mask]
        print(f"After adult filter: {len(df)} receptors")

    # Filter 2: Mapped to FlyWire
    mapped_mask = df['is_mapped'].astype(str).str.strip().str.lower() == 'yes'
    df = df[mapped_mask]
    print(f"After mapped filter: {len(df)} receptors")

    # Filter 3 & 4: Check ambiguity and ORN_* targets in mapping
    # Need to cross-reference with mapping CSV for ambiguity info
    door_names = df['door_name'].values
    valid_receptors = []

    for door_name in door_names:
        # Find corresponding FlyWire target in mapping
        mapping_rows = mapping_df[mapping_df['door_name'] == door_name]

        if len(mapping_rows) == 0:
            # Not in mapping at all
            continue

        # Get first non-ambiguous ORN_* target
        for _, row in mapping_rows.iterrows():
            target = str(row['flywire_glomerulus']).strip()
            if target.startswith('ORN_'):
                is_ambig = str(row.get('is_ambiguous', 'No')).strip().lower() in ['yes', 'true', '1']
                if not is_ambig or include_ambiguous:
                    valid_receptors.append(door_name)
                    break

    df = df[df['door_name'].isin(valid_receptors)]
    print(f"After ambiguity + ORN_* validation: {len(df)} receptors")

    return df


def extract_flywire_targets(
    receptor_names: list,
    mapping_df: pd.DataFrame,
) -> list:
    """Extract corresponding FlyWire ORN_* targets for receptors."""
    targets = []
    for receptor in receptor_names:
        mapping_rows = mapping_df[mapping_df['door_name'] == receptor]
        # Get first valid ORN_* target
        for _, row in mapping_rows.iterrows():
            target = str(row['flywire_glomerulus']).strip()
            if target.startswith('ORN_'):
                is_ambig = str(row.get('is_ambiguous', 'No')).strip().lower() in ['yes', 'true', '1']
                if not is_ambig:
                    targets.append(target)
                    break
    return targets


def collect_exclusions(
    full_inventory: pd.DataFrame,
    training_receptors: list,
    mapping_df: pd.DataFrame,
) -> dict:
    """Collect receptors excluded from training set with reasons."""
    # Rename receptor_name to door_name for consistency
    inv = full_inventory.copy()
    if 'receptor_name' in inv.columns:
        inv = inv.rename(columns={'receptor_name': 'door_name'})

    all_receptors = set(inv['door_name'].values)
    training_set = set(training_receptors)
    excluded = all_receptors - training_set

    # Categorize exclusions
    excluded_larval = []
    excluded_unmapped = []
    excluded_ambiguous = []

    for receptor in excluded:
        row = inv[inv['door_name'] == receptor].iloc[0]
        life_stage = str(row['life_stage']).strip()
        is_mapped = str(row['is_mapped']).strip().lower()

        # Check ambiguity in mapping
        is_ambig = False
        mapping_rows = mapping_df[mapping_df['door_name'] == receptor]
        if len(mapping_rows) > 0:
            is_ambig = mapping_rows.iloc[0].get('is_ambiguous', 'No')
            is_ambig = str(is_ambig).strip().lower() in ['yes', 'true', '1']

        if life_stage != 'Adult':
            excluded_larval.append(receptor)
        elif is_mapped != 'yes':
            excluded_unmapped.append(receptor)
        elif is_ambig:
            excluded_ambiguous.append(receptor)
        else:
            # Shouldn't happen, but log it
            excluded_unmapped.append(receptor)

    return {
        'excluded_larval': sorted(excluded_larval),
        'excluded_unmapped': sorted(excluded_unmapped),
        'excluded_ambiguous': sorted(excluded_ambiguous),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build training receptor set (adult + mapped + non-ambiguous)'
    )
    parser.add_argument(
        '--include_ambiguous',
        action='store_true',
        help='Include ambiguous mappings in training set',
    )
    parser.add_argument(
        '--allow_larval',
        action='store_true',
        help='Allow larval receptors (NOT RECOMMENDED)',
    )
    args = parser.parse_args()

    print("=" * 70)
    print("BUILDING TRAINING RECEPTOR SET")
    print("=" * 70)
    print()

    # Load data
    print(f"Loading inventory from {INVENTORY_FILE}...")
    inventory_df = load_inventory(INVENTORY_FILE)
    print(f"✓ Loaded {len(inventory_df)} receptors\n")

    print(f"Loading mapping from {MAPPING_FILE}...")
    mapping_df = load_mapping(MAPPING_FILE)
    print(f"✓ Loaded {len(mapping_df)} mapping rows\n")

    # Filter for training set
    print("Applying filters:")
    print(f"  - Adult only: {not args.allow_larval}")
    print(f"  - Mapped only: Yes")
    print(f"  - Exclude ambiguous: {not args.include_ambiguous}")
    print()

    training_df = filter_training_receptors(
        inventory_df,
        mapping_df,
        include_ambiguous=args.include_ambiguous,
        strict_adult_only=not args.allow_larval,
    )

    # Sort receptors alphabetically for consistency
    training_df = training_df.sort_values('door_name').reset_index(drop=True)
    receptor_names = training_df['door_name'].tolist()
    flywire_targets = extract_flywire_targets(receptor_names, mapping_df)

    assert len(receptor_names) == len(flywire_targets), \
        "Receptor count mismatch with FlyWire targets"

    # Collect exclusions
    exclusions = collect_exclusions(inventory_df, receptor_names, mapping_df)

    print()
    print("=" * 70)
    print("TRAINING RECEPTOR SET SUMMARY")
    print("=" * 70)
    print(f"Total receptors in inventory:     {len(inventory_df)}")
    print(f"Training receptors:               {len(receptor_names)}")
    print(f"  - Excluded (larval):            {len(exclusions['excluded_larval'])}")
    print(f"  - Excluded (unmapped):          {len(exclusions['excluded_unmapped'])}")
    print(f"  - Excluded (ambiguous):         {len(exclusions['excluded_ambiguous'])}")
    print()

    # Load connectivity metadata to create index mapping
    connectivity_metadata_path = Path("data/pgcn_features/connectivity/connectivity_metadata.json")
    connectivity_receptor_order = None
    receptor_indices_in_connectivity = None

    if connectivity_metadata_path.exists():
        with open(connectivity_metadata_path) as f:
            conn_meta = json.load(f)
            connectivity_receptor_order = conn_meta.get('receptor_names', [])

        # Create index mapping: training receptors → indices in connectivity matrices
        receptor_indices_in_connectivity = []
        missing_in_connectivity = []

        for receptor in receptor_names:
            try:
                idx = connectivity_receptor_order.index(receptor)
                receptor_indices_in_connectivity.append(idx)
            except ValueError:
                missing_in_connectivity.append(receptor)
                print(f"WARNING: {receptor} not found in connectivity metadata")

        if missing_in_connectivity:
            print(f"ERROR: {len(missing_in_connectivity)} training receptors missing from connectivity")
            print(f"Missing: {missing_in_connectivity}")
            raise ValueError("Training receptor set incompatible with connectivity matrices")

        print(f"\n✓ All training receptors found in connectivity matrices")
        print(f"  Training set size: {len(receptor_names)}")
        print(f"  Connectivity full size: {len(connectivity_receptor_order)}")
        print(f"  Index mapping: {len(receptor_indices_in_connectivity)} receptors")

    # Create training set artifact
    training_set = {
        'receptors': receptor_names,
        'flywire_targets': flywire_targets,
        'n_receptors': len(receptor_names),
        'receptor_indices_in_connectivity': receptor_indices_in_connectivity,
        'connectivity_receptor_order': connectivity_receptor_order,
        'filters': {
            'adult_only': not args.allow_larval,
            'mapped_only': True,
            'exclude_ambiguous': not args.include_ambiguous,
        },
        'excluded_larval': exclusions['excluded_larval'],
        'excluded_unmapped': exclusions['excluded_unmapped'],
        'excluded_ambiguous': exclusions['excluded_ambiguous'],
        'provenance': {
            'inventory_file': str(INVENTORY_FILE),
            'inventory_hash': hash_file(INVENTORY_FILE),
            'mapping_file': str(MAPPING_FILE),
            'mapping_hash': hash_file(MAPPING_FILE),
            'connectivity_metadata_file': str(connectivity_metadata_path) if connectivity_metadata_path.exists() else None,
            'timestamp': datetime.now().isoformat(),
        },
    }

    # Save JSON
    print(f"Saving training set JSON to {OUTPUT_JSON}...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(training_set, f, indent=2)
    print(f"✓ Saved {OUTPUT_JSON}")

    # Save CSV (human-readable)
    print(f"Saving training set CSV to {OUTPUT_CSV}...")
    csv_df = pd.DataFrame({
        'receptor': receptor_names,
        'flywire_target': flywire_targets,
    })
    csv_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved {OUTPUT_CSV}")

    print()
    print("=" * 70)
    print("✅ TRAINING RECEPTOR SET BUILT SUCCESSFULLY")
    print("=" * 70)
    print()
    print(f"Training receptors: {len(receptor_names)}")
    print(f"First 10: {receptor_names[:10]}")
    print(f"Last 10:  {receptor_names[-10:]}")
    print()


if __name__ == '__main__':
    main()
