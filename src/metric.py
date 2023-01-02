import argparse
import sys
sys.path.append('.')

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


import scipy.spatial as spatial


def diameter(data):
    if data.shape[0] <= 1:
        return 0
    if data.shape[0] == 2:
        return ((data[0] - data[1])**2).sum()
    hull = spatial.ConvexHull(data)
    candidates = data[hull.vertices]
    return np.max(pairwise_distances(candidates))

def compactness_score(data, pred):
    n_labels = np.max(pred) + 1
    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(data[0])), dtype=float)
    for k in range(n_labels):
        cluster_k = data[pred == k]
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]))
    return np.mean(intra_dists)

def separation_score(data, pred):
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

def culc_internal_metrics(data, pred):
    metrics = {}
    metrics['compactness'] = compactness_score(data, pred)
    metrics['separation'] = separation_score(data, pred)
    metrics['SSE'] = SSE(data, pred)
    metrics['silhouette coefficient'] = silhouette_score(data, pred)
    metrics['calinski harabasz score'] = calinski_harabasz_score(data, pred)
    metrics['davies bouldin index'] = davies_bouldin_score(data, pred)
    # O(n^2)
    metrics['dunn validity index'] = dunn_validity_index_score(data, pred)

    return metrics


def main(args):
    x, y = make_blobs(n_samples=500, n_features=3, centers=4, random_state=1)

    cluster = KMeans(n_clusters=4, random_state=0)
    print(cluster)
    cluster.fit(x)

    internal_metrics = culc_internal_metrics(x, cluster.labels_)
    for k, v in internal_metrics.items():
        print(k, v)
    for i in range(4):
        plt.scatter(x[y==i, 0], x[y==i, 1])
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
