#!/usr/bin/env python3
"""
Basic DoOR Toolkit Usage Example
=================================

Demonstrates extraction and encoding of DoOR data.
"""

from door_toolkit import DoORExtractor, DoOREncoder
from door_toolkit.utils import list_odorants, find_similar_odorants


def main():
    print("="*70)
    print("DoOR Python Toolkit - Basic Example")
    print("="*70)
    
    # Step 1: Extract DoOR data (run once)
    print("\n[1] Extracting DoOR data...")
    print("    (Skip this if you already have a cache)")
    
    # Uncomment to run extraction:
    # extractor = DoORExtractor(
    #     input_dir="path/to/DoOR.data/data",
    #     output_dir="door_cache"
    # )
    # extractor.run()
    
    # Step 2: Load encoder
    print("\n[2] Loading encoder...")
    encoder = DoOREncoder("door_cache")
    print(f"    Loaded {len(encoder.odorant_names)} odorants")
    print(f"    {encoder.n_channels} receptor channels")
    
    # Step 3: Encode odorants
    print("\n[3] Encoding odorants...")
    test_odors = ["acetic acid", "1-pentanol", "ethyl acetate"]
    
    for odor in test_odors:
        try:
            pn = encoder.encode(odor)
            stats = encoder.get_receptor_coverage(odor)
            print(f"    {odor}:")
            print(f"      - PN shape: {pn.shape}")
            print(f"      - Active receptors: {stats['n_active']}/{stats['n_tested']}")
            print(f"      - Max response: {stats['max_response']:.3f}")
        except KeyError:
            print(f"    {odor}: NOT FOUND")
    
    # Step 4: Batch encoding
    print("\n[4] Batch encoding...")
    pn_batch = encoder.batch_encode(test_odors[:2])  # First 2 found
    print(f"    Batch shape: {pn_batch.shape}")
    
    # Step 5: Search odorants
    print("\n[5] Searching for acetates...")
    acetates = encoder.list_available_odorants(pattern="acetate")
    print(f"    Found {len(acetates)} acetates")
    print(f"    Examples: {acetates[:5]}")
    
    # Step 6: Find similar odorants
    print("\n[6] Finding similar odorants to 'acetic acid'...")
    similar = find_similar_odorants(
        "acetic acid",
        "door_cache",
        top_k=5,
        method="correlation"
    )
    for name, similarity in similar:
        print(f"    {name}: {similarity:.3f}")
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)


if __name__ == "__main__":
    main()
