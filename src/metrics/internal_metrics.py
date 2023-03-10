import numpy as np

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances

def compactness(data, pred):
    n_labels = np.max(pred) + 1
    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(data[0])), dtype=float)
    for k in range(n_labels):
        cluster_k = data[pred == k]
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]))
    return np.mean(intra_dists)


def separation(data, pred):
    n_labels = np.max(pred) + 1
    centroids = np.zeros((n_labels, len(data[0])), dtype=float)
    for k in range(n_labels):
        cluster_k = data[pred == k]
        centroids[k] = cluster_k.mean(axis=0)
    centroid_distances = pairwise_distances(centroids)
    return np.sum(centroid_distances) / (n_labels * (n_labels - 1))


def dunn_validity_index_score(data, pred):
    n_labels = np.max(pred) + 1
    max_intra_dists = np.zeros(n_labels)
    min_extra_dists = np.full((n_labels, n_labels), np.inf)

    dists = pairwise_distances(data)
    
    for k in range(n_labels):
        pos_k = pred == k
        max_intra_dists[k] = np.max(dists[pos_k][:, pos_k])

    for i in range(n_labels):
        pos_i = pred == i
        for j in range(i):
            dists_ij = dists[pos_i][:, pred == j]
            min_extra_dists[i, j] = np.min(dists_ij[np.nonzero(dists_ij)])
    
    return np.min(min_extra_dists) / np.max(max_intra_dists)


def SSE(data, pred):
    n_labels = np.max(pred) + 1
    sse = 0
    for k in range(n_labels):
        cluster_k = data[pred == k]
        centroid = cluster_k.mean(axis=0)
        sse += np.sum((cluster_k - centroid) ** 2)
    return sse