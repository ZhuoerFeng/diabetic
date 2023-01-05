import argparse
import sys
sys.path.append('.')

from metrics.internal_metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, \
                                            dunn_validity_index_score, compactness, separation, SSE
from metrics.cluster_tendency import hopkins_statistics

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

def culc_internal_metrics(data, pred):
    metrics = {}
    metrics['compactness'] = compactness(data, pred)
    metrics['separation'] = separation(data, pred)
    metrics['SSE'] = SSE(data, pred)
    metrics['silhouette coefficient'] = silhouette_score(data, pred)
    metrics['calinski harabasz score'] = calinski_harabasz_score(data, pred)
    metrics['davies bouldin index'] = davies_bouldin_score(data, pred)
    # O(n^2)
    metrics['dunn validity index'] = dunn_validity_index_score(data, pred)

    return metrics


def culc_cluster_tendency(data):
    metrics = {}
    metrics['hopkins_statistics'] = hopkins_statistics(data, sample_ratio=0.3)
    return metrics

def main(args):
    # data = pd.read_csv('data/diabetic_data_lht.csv').iloc[:, 2:-1]
    # # print(data)
    # scaler = MinMaxScaler()
    # data = scaler.fit_transform(data)
    # print(culc_cluster_tendency(data))

    x, y = make_blobs(n_samples=500, n_features=3, centers=4, random_state=1)
    
    cluster_tendency = culc_cluster_tendency(x)
    print('cluster tendency:')
    for k, v in cluster_tendency.items():
        print(k + ':', v)
    cluster = KMeans(n_clusters=4, random_state=0)
    cluster.fit(x)

    internal_metrics = culc_internal_metrics(x, cluster.labels_)
    print('\ninternal metrics:')
    for k, v in internal_metrics.items():
        print(k + ':', v)
    for i in range(4):
        plt.scatter(x[y==i, 0], x[y==i, 1])
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
