"""
Pathway Analysis Example
=========================

This example demonstrates pathway tracing, Shapley analysis, and
experiment protocol generation for olfactory circuits.

Requirements:
    - DoOR cache (run door-extract first)
"""

from door_toolkit.pathways import (
    PathwayAnalyzer,
    BlockingExperimentGenerator,
    BehavioralPredictor,
)

# Example 1: Trace Or47b feeding pathway
print("=" * 70)
print("Example 1: Or47b → Hexanol → Feeding Pathway")
print("=" * 70)

analyzer = PathwayAnalyzer("door_cache")
pathway = analyzer.trace_or47b_feeding_pathway()

print(f"\nPathway: {pathway.pathway_name}")
print(f"Target Behavior: {pathway.target_behavior}")
print(f"Pathway Strength: {pathway.strength:.3f}")

print(f"\nReceptor Contributions:")
for receptor, contrib in pathway.get_top_receptors():
    print(f"  {receptor}: {contrib:.3f}")

print(f"\nMetadata:")
for key, value in pathway.metadata.items():
    print(f"  {key}: {value}")


# Example 2: Trace Or42b pathway
print("\n" + "=" * 70)
print("Example 2: Or42b Fruit Ester Pathway")
print("=" * 70)

or42b_pathway = analyzer.trace_or42b_pathway()

print(f"\nPathway: {or42b_pathway.pathway_name}")
print(f"Strength: {or42b_pathway.strength:.3f}")


# Example 3: Custom pathway analysis
print("\n" + "=" * 70)
print("Example 3: Custom Pathway (Or92a → Geosmin → Avoidance)")
print("=" * 70)

custom_pathway = analyzer.trace_custom_pathway(
    receptors=["Or92a"],
    odorants=["geosmin"],
    behavior="avoidance",
)

print(f"\nPathway: {custom_pathway.pathway_name}")
print(f"Strength: {custom_pathway.strength:.3f}")


# Example 4: Compute Shapley importance
print("\n" + "=" * 70)
print("Example 4: Shapley Importance for Feeding Behavior")
print("=" * 70)

print("\nComputing Shapley values (this may take a moment)...")
importance = analyzer.compute_shapley_importance("feeding")

print(f"\nTop 10 receptors by importance:")
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
for receptor, score in sorted_importance[:10]:
    print(f"  {receptor}: {score:.4f}")


# Example 5: Find critical blocking targets
print("\n" + "=" * 70)
print("Example 5: Critical Blocking Targets")
print("=" * 70)

targets = analyzer.find_critical_blocking_targets(pathway, threshold=0.1)

print(f"\nCritical blocking targets (threshold=0.1):")
for target in targets:
    print(f"  - {target}")


# Example 6: Generate experiment protocol
print("\n" + "=" * 70)
print("Example 6: Generate Experiment Protocol")
print("=" * 70)

generator = BlockingExperimentGenerator("door_cache")

# Generate single-unit veto experiment
protocol = generator.generate_experiment_1_protocol()

print(f"\nExperiment: {protocol.experiment_name}")
print(f"ID: {protocol.experiment_id}")
print(f"Hypothesis: {protocol.hypothesis}")

print(f"\nExperimental Steps ({len(protocol.steps)} total):")
for step in protocol.steps[:3]:  # Show first 3
    print(f"  Step {step.step_number}: {step.action}")
    print(f"    Target: {step.target}")
    print(f"    Method: {step.method}")

print(f"\nControls ({len(protocol.controls)} total):")
for control in protocol.controls[:3]:
    print(f"  - {control}")

# Export protocol
protocol.export_json("output/experiment_1_protocol.json")
protocol.export_markdown("output/experiment_1_protocol.md")
print(f"\nExported protocol to output/")


# Example 7: Behavioral prediction
print("\n" + "=" * 70)
print("Example 7: Behavioral Prediction")
print("=" * 70)

predictor = BehavioralPredictor("door_cache")

# Predict behavior for different odorants
odorants = ["hexanol", "geosmin", "ethyl butyrate"]

print(f"\nBehavioral predictions:")
for odorant in odorants:
    try:
        prediction = predictor.predict_behavior(odorant)
        print(f"\n  {odorant}:")
        print(f"    Valence: {prediction.predicted_valence}")
        print(f"    Confidence: {prediction.confidence:.2%}")
        print(f"    Top receptors: {', '.join([r for r, _ in prediction.key_contributors[:3]])}")
    except Exception as e:
        print(f"\n  {odorant}: Could not predict ({e})")


# Example 8: Compare pathways
print("\n" + "=" * 70)
print("Example 8: Pathway Comparison")
print("=" * 70)

all_pathways = [pathway, or42b_pathway, custom_pathway]
comparison = analyzer.compare_pathways(all_pathways)

print(f"\nPathway Comparison:")
print(comparison.to_string(index=False))

print("\nDone! Check the output/ directory for exported files.")
