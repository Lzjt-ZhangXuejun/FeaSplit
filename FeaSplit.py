import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class FeaSplit:
    def __init__(self, num_clients):
        self.n_clusters = num_clients
        self.subspaces = [None] * num_clients
        self.centroids = None

    def _compute_subspace(self, X):
        svd = TruncatedSVD(n_components=10)
        return svd.fit(X).components_.T

    def _principal_angle_penalty(self, X, cluster_id):
        if self.subspaces[cluster_id] is None:
            return 0
        proj = X @ self.subspaces[cluster_id]
        return np.linalg.norm(X - proj @ self.subspaces[cluster_id].T, axis=1)

    def fit(self, X, max_iter=100):
        # global new_labels
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=2025)
        initial_labels = kmeans.fit_predict(X)
        self.centroids = kmeans.cluster_centers_

        for k in range(self.n_clusters):
            cluster_data = X[initial_labels == k]
            if len(cluster_data) > 10:
                self.subspaces[k] = self._compute_subspace(cluster_data)

        for _ in range(max_iter):
            distances = cdist(X, self.centroids, 'euclidean')
            for k in range(self.n_clusters):
                if self.subspaces[k] is not None:
                    penalties = self._principal_angle_penalty(X, k)
                    distances[:, k] += penalties

            new_labels = np.argmin(distances, axis=1)
            new_centroids = np.zeros_like(self.centroids)

            for k in range(self.n_clusters):
                cluster_data = X[new_labels == k]
                if len(cluster_data) > 0:
                    new_centroids[k] = cluster_data.mean(axis=0)
                    if len(cluster_data) > 10:
                        self.subspaces[k] = self._compute_subspace(cluster_data)
                else:
                    self.subspaces[k] = None

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return new_labels

    def compute_principal_angle_matrix(self):
        """
        计算各客户端子空间之间的平均主角度矩阵
        返回: (n_clusters, n_clusters) 的 numpy 矩阵
        """
        n = self.n_clusters
        angle_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    angle_matrix[i, j] = 0.0
                elif self.subspaces[i] is not None and self.subspaces[j] is not None:
                    M = self.subspaces[i].T @ self.subspaces[j]
                    s = np.linalg.svd(M, compute_uv=False)
                    angles = np.arccos(np.clip(s, -1.0, 1.0))
                    angle_matrix[i, j] = np.mean(angles)
                else:
                    angle_matrix[i, j] = np.nan

        return angle_matrix
