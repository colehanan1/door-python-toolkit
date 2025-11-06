"""
Spatial Analysis Module
========================

3D spatial activation mapping and visualization for FlyWire-DoOR integration.

This module provides tools for analyzing spatial patterns of olfactory receptor
activation in the Drosophila brain.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SpatialActivationMap:
    """
    3D spatial activation map with visualization capabilities.

    Attributes:
        odorant_name: Name of the odorant
        points: List of (x, y, z, intensity) tuples
        receptor_data: Mapping of receptor names to activation strengths
        metadata: Additional metadata about the map
    """

    odorant_name: str
    points: List[Tuple[float, float, float, float]]
    receptor_data: Dict[str, float]
    metadata: Optional[Dict] = None

    @property
    def n_points(self) -> int:
        """Number of spatial points."""
        return len(self.points)

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Get spatial bounds (min, max) for each dimension."""
        if not self.points:
            return ((0, 0), (0, 0), (0, 0))

        coords = np.array(self.points)
        return (
            (coords[:, 0].min(), coords[:, 0].max()),
            (coords[:, 1].min(), coords[:, 1].max()),
            (coords[:, 2].min(), coords[:, 2].max()),
        )

    @property
    def mean_activation(self) -> float:
        """Mean activation intensity across all points."""
        if not self.points:
            return 0.0
        return float(np.mean([p[3] for p in self.points]))

    @property
    def max_activation(self) -> float:
        """Maximum activation intensity."""
        if not self.points:
            return 0.0
        return float(np.max([p[3] for p in self.points]))

    def get_centroid(self) -> Tuple[float, float, float]:
        """
        Calculate the centroid of activation.

        Returns:
            (x, y, z) coordinates of weighted centroid
        """
        if not self.points:
            return (0.0, 0.0, 0.0)

        coords = np.array(self.points)
        intensities = coords[:, 3]

        # Weight by activation intensity
        total_intensity = intensities.sum()
        if total_intensity == 0:
            # Unweighted centroid
            return tuple(coords[:, :3].mean(axis=0).tolist())

        weighted_coords = coords[:, :3] * intensities[:, np.newaxis]
        centroid = weighted_coords.sum(axis=0) / total_intensity

        return tuple(centroid.tolist())

    def filter_by_threshold(self, threshold: float) -> "SpatialActivationMap":
        """
        Create new map with only points above threshold.

        Args:
            threshold: Minimum activation intensity

        Returns:
            New SpatialActivationMap with filtered points
        """
        filtered_points = [p for p in self.points if p[3] >= threshold]

        # Filter receptor data as well
        filtered_receptors = {k: v for k, v in self.receptor_data.items() if v >= threshold}

        return SpatialActivationMap(
            odorant_name=self.odorant_name,
            points=filtered_points,
            receptor_data=filtered_receptors,
            metadata=self.metadata,
        )

    def cluster_analysis(
        self, n_clusters: int = 3, method: str = "kmeans"
    ) -> Dict[int, List[Tuple[float, float, float, float]]]:
        """
        Perform spatial clustering on activation points.

        Args:
            n_clusters: Number of clusters
            method: Clustering method ('kmeans' or 'hierarchical')

        Returns:
            Dictionary mapping cluster ID to list of points

        Raises:
            ImportError: If scikit-learn not available
        """
        if not self.points:
            return {}

        try:
            from sklearn.cluster import KMeans, AgglomerativeClustering
        except ImportError:
            raise ImportError(
                "scikit-learn required for clustering. "
                "Install with: pip install door-python-toolkit[flywire]"
            )

        coords = np.array([p[:3] for p in self.points])

        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        labels = clusterer.fit_predict(coords)

        # Group points by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.points[i])

        logger.info(f"Clustered {len(self.points)} points into {n_clusters} clusters")
        return clusters

    def export_csv(self, output_path: str) -> None:
        """
        Export spatial map to CSV file.

        Args:
            output_path: Output file path

        Example:
            >>> spatial_map.export_csv("activation_map.csv")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for x, y, z, intensity in self.points:
            rows.append(
                {
                    "odorant": self.odorant_name,
                    "x": x,
                    "y": y,
                    "z": z,
                    "intensity": intensity,
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(rows)} points to {output_path}")

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        return {
            "odorant_name": self.odorant_name,
            "n_points": self.n_points,
            "points": [{"x": x, "y": y, "z": z, "intensity": i} for x, y, z, i in self.points],
            "receptor_data": self.receptor_data,
            "statistics": {
                "mean_activation": self.mean_activation,
                "max_activation": self.max_activation,
                "centroid": self.get_centroid(),
                "bounds": {
                    "x": self.bounds[0],
                    "y": self.bounds[1],
                    "z": self.bounds[2],
                },
            },
            "metadata": self.metadata,
        }


def compare_spatial_maps(
    maps: List[SpatialActivationMap],
) -> pd.DataFrame:
    """
    Compare multiple spatial activation maps.

    Args:
        maps: List of SpatialActivationMap objects

    Returns:
        DataFrame with comparison statistics

    Example:
        >>> map1 = SpatialActivationMap(...)
        >>> map2 = SpatialActivationMap(...)
        >>> comparison = compare_spatial_maps([map1, map2])
        >>> print(comparison)
    """
    rows = []
    for spatial_map in maps:
        centroid = spatial_map.get_centroid()
        row = {
            "odorant": spatial_map.odorant_name,
            "n_points": spatial_map.n_points,
            "mean_activation": spatial_map.mean_activation,
            "max_activation": spatial_map.max_activation,
            "centroid_x": centroid[0],
            "centroid_y": centroid[1],
            "centroid_z": centroid[2],
            "n_receptors": len(spatial_map.receptor_data),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_spatial_overlap(
    map1: SpatialActivationMap,
    map2: SpatialActivationMap,
    radius: float = 1000.0,
) -> float:
    """
    Calculate spatial overlap between two activation maps.

    Args:
        map1: First activation map
        map2: Second activation map
        radius: Distance threshold for considering points overlapping (nm)

    Returns:
        Overlap score between 0 and 1

    Example:
        >>> overlap = calculate_spatial_overlap(map1, map2, radius=500.0)
        >>> print(f"Overlap: {overlap:.2%}")
    """
    if not map1.points or not map2.points:
        return 0.0

    coords1 = np.array([p[:3] for p in map1.points])
    coords2 = np.array([p[:3] for p in map2.points])

    # Count points from map1 that have a nearby point in map2
    overlapping = 0
    for point1 in coords1:
        distances = np.sqrt(((coords2 - point1) ** 2).sum(axis=1))
        if np.any(distances < radius):
            overlapping += 1

    overlap_score = overlapping / len(coords1)
    logger.debug(
        f"Spatial overlap between '{map1.odorant_name}' and '{map2.odorant_name}': "
        f"{overlap_score:.2%} (radius={radius}nm)"
    )

    return overlap_score


def create_activation_heatmap(
    spatial_map: SpatialActivationMap,
    resolution: int = 50,
    dimension: str = "xy",
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Create 2D heatmap projection of 3D activation.

    Args:
        spatial_map: Spatial activation map
        resolution: Grid resolution for heatmap
        dimension: Projection plane ('xy', 'xz', or 'yz')

    Returns:
        Tuple of (heatmap_array, x_range, y_range)

    Example:
        >>> heatmap, x_range, y_range = create_activation_heatmap(map, dimension="xy")
        >>> plt.imshow(heatmap, extent=[*x_range, *y_range])
    """
    if not spatial_map.points:
        return np.zeros((resolution, resolution)), (0, 1), (0, 1)

    coords = np.array(spatial_map.points)

    # Select dimensions based on projection
    if dimension == "xy":
        x_idx, y_idx, intensity_idx = 0, 1, 3
    elif dimension == "xz":
        x_idx, y_idx, intensity_idx = 0, 2, 3
    elif dimension == "yz":
        x_idx, y_idx, intensity_idx = 1, 2, 3
    else:
        raise ValueError(f"Unknown dimension: {dimension}")

    x_coords = coords[:, x_idx]
    y_coords = coords[:, y_idx]
    intensities = coords[:, intensity_idx]

    # Create grid
    x_range = (x_coords.min(), x_coords.max())
    y_range = (y_coords.min(), y_coords.max())

    heatmap, x_edges, y_edges = np.histogram2d(
        x_coords,
        y_coords,
        bins=resolution,
        range=[x_range, y_range],
        weights=intensities,
    )

    # Normalize counts
    counts, _, _ = np.histogram2d(x_coords, y_coords, bins=resolution, range=[x_range, y_range])
    counts[counts == 0] = 1  # Avoid division by zero
    heatmap = heatmap / counts

    return heatmap.T, x_range, y_range


def visualize_spatial_map(
    spatial_map: SpatialActivationMap,
    output_path: Optional[str] = None,
    colormap: str = "viridis",
) -> None:
    """
    Visualize spatial activation map using matplotlib.

    Args:
        spatial_map: Spatial activation map to visualize
        output_path: Optional path to save figure
        colormap: Matplotlib colormap name

    Example:
        >>> visualize_spatial_map(map, output_path="activation_map.png")

    Raises:
        ImportError: If matplotlib not available
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        raise ImportError(
            "matplotlib required for visualization. "
            "Install with: pip install door-python-toolkit[flywire]"
        )

    if not spatial_map.points:
        logger.warning("No points to visualize")
        return

    coords = np.array(spatial_map.points)

    fig = plt.figure(figsize=(15, 5))

    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection="3d")
    scatter = ax1.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=coords[:, 3],
        cmap=colormap,
        s=50,
        alpha=0.6,
    )
    ax1.set_xlabel("X (nm)")
    ax1.set_ylabel("Y (nm)")
    ax1.set_zlabel("Z (nm)")
    ax1.set_title(f"3D Activation: {spatial_map.odorant_name}")
    plt.colorbar(scatter, ax=ax1, label="Activation")

    # XY projection heatmap
    ax2 = fig.add_subplot(132)
    heatmap, x_range, y_range = create_activation_heatmap(spatial_map, dimension="xy")
    im = ax2.imshow(
        heatmap,
        extent=[*x_range, *y_range],
        origin="lower",
        cmap=colormap,
        aspect="auto",
    )
    ax2.set_xlabel("X (nm)")
    ax2.set_ylabel("Y (nm)")
    ax2.set_title("XY Projection")
    plt.colorbar(im, ax=ax2, label="Mean Activation")

    # XZ projection heatmap
    ax3 = fig.add_subplot(133)
    heatmap, x_range, z_range = create_activation_heatmap(spatial_map, dimension="xz")
    im = ax3.imshow(
        heatmap,
        extent=[*x_range, *z_range],
        origin="lower",
        cmap=colormap,
        aspect="auto",
    )
    ax3.set_xlabel("X (nm)")
    ax3.set_ylabel("Z (nm)")
    ax3.set_title("XZ Projection")
    plt.colorbar(im, ax=ax3, label="Mean Activation")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()
