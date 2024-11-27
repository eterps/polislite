#!/usr/bin/env python3
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from polis_core import OpinionAnalyzer


def load_from_yaml(filepath):
    with open(filepath) as f:
        data = yaml.safe_load(f)
    vote_map = {"agree": 1, "disagree": -1}
    votes = [
        [vote_map.get(v, 0) for v in user_votes]
        for user_votes in data["votes"].values()
    ]
    return data["statements"], np.array(votes), list(data["votes"].keys())


def plot_opinion_clusters(points_2d, clusters, usernames, output_path=None):
    plt.figure(figsize=(12, 8))

    # Draw buffered convex hull for each cluster
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        cluster_points = points_2d[mask]

        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]

            # Buffer the hull
            centroid = np.mean(hull_points, axis=0)
            vectors = hull_points - centroid
            lengths = np.sqrt(np.sum(vectors**2, axis=1))
            normalized_vectors = vectors / lengths[:, np.newaxis]
            buffered_points = hull_points + normalized_vectors * 0.5
            buffered_points = np.vstack((buffered_points, buffered_points[0]))

            # Draw hull
            color = plt.cm.viridis(cluster_id / len(np.unique(clusters)))
            plt.gca().add_patch(Polygon(buffered_points, alpha=0.2, facecolor=color))

    # Plot points and labels
    scatter = plt.scatter(
        points_2d[:, 0], points_2d[:, 1], c=clusters, cmap="viridis", s=100, alpha=0.6
    )

    for i, user in enumerate(usernames):
        plt.annotate(
            user,
            (points_2d[i, 0], points_2d[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Set view limits with padding
    x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
    y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
    padding_x = (x_max - x_min) * 0.2
    padding_y = (y_max - y_min) * 0.2
    plt.xlim(x_min - padding_x, x_max + padding_x)
    plt.ylim(y_min - padding_y, y_max + padding_y)

    plt.title("Opinion Clusters")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.7)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


def main(yaml_file):
    statements, votes, usernames = load_from_yaml(yaml_file)
    analyzer = OpinionAnalyzer()
    results = analyzer.analyze(votes, statements)

    # Add jitter to separate overlapping points
    points_2d = results["points_2d"]
    jitter = np.random.normal(0, 0.1, points_2d.shape)
    jittered_points = points_2d + jitter

    output_path = Path(yaml_file).with_suffix(".png")
    plot_opinion_clusters(jittered_points, results["clusters"], usernames, output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python plot_opinions.py input.yaml")
        sys.exit(1)
    main(sys.argv[1])
