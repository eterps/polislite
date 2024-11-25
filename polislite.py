import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score
from collections import defaultdict

class PolisClusterer:
    def __init__(self, min_clusters=2, max_clusters=6):
        self.pca = PCA(n_components=2)
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        
    def analyze_opinions(self, votes, statements):
        vote_matrix = np.array([
            [1 if v == 'agree' else -1 if v == 'disagree' else 0 
             for v in voter_votes] 
            for voter_votes in votes
        ])
        
        self._handle_sparse_votes(vote_matrix)
        points_2d = self._compute_pca(vote_matrix)
        clusters = self._find_optimal_clusters(points_2d)
        
        self._generate_report(vote_matrix, clusters, statements)
        return points_2d, clusters
    
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
        
        # Compute average distance between cluster centers
        diffs = []
        for i in cluster_means:
            for j in cluster_means:
                if i < j:
                    diff = np.linalg.norm(cluster_means[i] - cluster_means[j])
                    diffs.append(diff)
        return np.mean(diffs) if diffs else 0

    def _find_optimal_clusters(self, points):
        linkage = hierarchy.linkage(points, method='ward')
        
        max_clusters = min(self.max_clusters, len(points) - 1)
        scores = []
        
        for n in range(self.min_clusters, max_clusters + 1):
            clusters = hierarchy.fcluster(linkage, t=n, criterion='maxclust')
            
            silhouette = silhouette_score(points, clusters) if len(np.unique(clusters)) > 1 else -1
            
            group_sizes = np.bincount(clusters)
            size_balance = np.min(group_sizes) / np.max(group_sizes)
            
            pattern_diff = self._compute_pattern_difference(clusters, points)
            
            score = (silhouette * 0.4 + size_balance * 0.3 + pattern_diff * 0.3)
            scores.append(score)
        
        optimal_n = self.min_clusters + np.argmax(scores)
        return hierarchy.fcluster(linkage, t=optimal_n, criterion='maxclust')
    
    def _generate_report(self, vote_matrix, clusters, statements):
        statement_scores = np.mean(vote_matrix, axis=0)
        agreement_levels = np.std(vote_matrix, axis=0)
        
        print('Consensus Statements:')
        for stmt, score, agree in zip(statements, statement_scores, agreement_levels):
            if agree < 0.5:
                consensus = 'strong agreement' if score > 0.5 else 'strong disagreement'
                print(f'- {stmt} ({consensus})')
        
        print('\nDivisive Statements:')
        for stmt, agree in zip(statements, agreement_levels):
            if agree >= 0.5:
                print(f'- {stmt}')
        
        cluster_opinions = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            cluster_opinions[cluster_id].append(vote_matrix[i])
            
        print('\nGroup Positions:')
        for grp_id in sorted(cluster_opinions.keys()):
            opinions = np.mean(cluster_opinions[grp_id], axis=0)
            print(f'\nGroup {grp_id} characteristics:')
            for stmt, opinion in zip(statements, opinions):
                if abs(opinion) > 0.5:
                    stance = 'strongly agrees with' if opinion > 0 else 'strongly disagrees with'
                    print(f'- {stance}: {stmt}')

# Example usage
statements = [
    'Climate change requires immediate action',
    'Nuclear power is necessary for clean energy',
    'Carbon tax should be implemented globally',
    'Individual actions matter for sustainability',
    'Companies should be held liable for emissions'
]

votes = [
    # Group 1: Environmental purists (anti-nuclear)
    ['agree', 'disagree', 'agree', 'agree', 'agree'],
    ['agree', 'disagree', 'agree', 'agree', 'agree'],
    ['agree', 'disagree', 'agree', 'agree', 'agree'],
    
    # Group 2: Tech-focused environmentalists (pro-nuclear)
    ['agree', 'agree', 'agree', 'disagree', 'agree'],
    ['agree', 'agree', 'agree', 'disagree', 'agree'],
    ['agree', 'agree', 'agree', 'disagree', 'agree'],
    
    # Group 3: Business-oriented (anti-regulation)
    ['agree', 'agree', 'disagree', 'disagree', 'disagree'],
    ['agree', 'agree', 'disagree', 'disagree', 'disagree'],
    ['agree', 'agree', 'disagree', 'disagree', 'disagree']
]

clusterer = PolisClusterer()
points, clusters = clusterer.analyze_opinions(votes, statements)
