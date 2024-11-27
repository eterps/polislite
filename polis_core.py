import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score
from collections import defaultdict


class OpinionAnalyzer:
    def __init__(self, min_clusters=2, max_clusters=6):
        self.pca = PCA(n_components=2)
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

    def analyze(self, vote_matrix, statements):
        """
        Analyze voting patterns and return structured results.

        Args:
            vote_matrix: numpy array of votes (1 for agree, -1 for disagree)
            statements: list of statement strings

        Returns:
            dict containing:
            - points_2d: 2D PCA projection of votes
            - clusters: cluster assignments
            - consensus_data: list of (statement, score, agreement_level)
            - divisive_data: list of (statement, agreement_level)
            - group_data: dict of group_id -> list of (statement, opinion)
        """
        self._handle_sparse_votes(vote_matrix)
        points_2d = self._compute_pca(vote_matrix)
        clusters = self._find_optimal_clusters(points_2d)

        # Calculate consensus and group data
        statement_scores = np.mean(vote_matrix, axis=0)
        agreement_levels = np.std(vote_matrix, axis=0)

        cluster_opinions = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            cluster_opinions[cluster_id].append(vote_matrix[i])

        group_data = {}
        for grp_id in sorted(cluster_opinions.keys()):
            opinions = np.mean(cluster_opinions[grp_id], axis=0)
            significant_opinions = [
                (stmt, opinion)
                for stmt, opinion in zip(statements, opinions)
                if abs(opinion) > 0.5
            ]
            group_data[grp_id] = significant_opinions

        return {
            "points_2d": points_2d,
            "clusters": clusters,
            "consensus_data": list(zip(statements, statement_scores, agreement_levels)),
            "divisive_data": list(zip(statements, agreement_levels)),
            "group_data": group_data,
        }

    def _handle_sparse_votes(self, matrix):
        row_means = np.nanmean(matrix, axis=1)
        for i, row in enumerate(matrix):
            matrix[i][row == 0] = row_means[i]

    def _compute_pca(self, matrix):
        masked_matrix = np.ma.masked_where(matrix == 0, matrix)
        return self.pca.fit_transform(masked_matrix)

    def _compute_pattern_difference(self, clusters, points):
        cluster_means = defaultdict(list)
        for i, cluster in enumerate(clusters):
            cluster_means[cluster].append(points[i])

        cluster_means = {k: np.mean(v, axis=0) for k, v in cluster_means.items()}

        diffs = []
        for i in cluster_means:
            for j in cluster_means:
                if i < j:
                    diff = np.linalg.norm(cluster_means[i] - cluster_means[j])
                    diffs.append(diff)
        return np.mean(diffs) if diffs else 0

    def _find_optimal_clusters(self, points):
        linkage = hierarchy.linkage(points, method="ward")

        max_clusters = min(self.max_clusters, len(points) - 1)
        scores = []

        for n in range(self.min_clusters, max_clusters + 1):
            clusters = hierarchy.fcluster(linkage, t=n, criterion="maxclust")

            silhouette = (
                silhouette_score(points, clusters)
                if len(np.unique(clusters)) > 1
                else -1
            )

            group_sizes = np.bincount(clusters)
            size_balance = np.min(group_sizes) / np.max(group_sizes)

            pattern_diff = self._compute_pattern_difference(clusters, points)

            score = silhouette * 0.4 + size_balance * 0.3 + pattern_diff * 0.3
            scores.append(score)

        optimal_n = self.min_clusters + np.argmax(scores)
        return hierarchy.fcluster(linkage, t=optimal_n, criterion="maxclust")
