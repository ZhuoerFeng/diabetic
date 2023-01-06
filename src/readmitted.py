from metrics import *
from sklearn import preprocessing
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np

def main(args):
    data = pd.read_csv(args.data_pred_file)
    truth = pd.read_csv('data/diabetic_data_lht.csv').iloc[:, -1].to_numpy()

    raw_data = data.iloc[:, 0:-1].values
    min_max_scaler = preprocessing.MinMaxScaler()
    # normalize for k-means distance calculation
    minmax_data = min_max_scaler.fit_transform(raw_data)
    normalized_data = pd.DataFrame(minmax_data).apply(pd.to_numeric).to_numpy()
    pred = data.iloc[:, -1].values
    n_clusters = np.max(pred) + 1
    print(n_clusters)
    xs = ['cluster' + str(i + 1) for i in range(n_clusters)]
    cluster_truths = [[truth[j] for j in range(len(pred)) if pred[j] == cluster_i] for cluster_i in range(n_clusters)]
    print([len(l) for l in cluster_truths])
    no_count = np.array([len([label for label in truths if label == 'NO']) / len(truths) for truths in cluster_truths])
    gt30_count = np.array([len([label for label in truths if label == '>30']) / len(truths) for truths in cluster_truths])
    lt30_count = np.array([len([label for label in truths if label == '<30']) / len(truths) for truths in cluster_truths])

    plt.bar(xs, lt30_count, label='<30')
    plt.bar(xs, gt30_count, bottom=lt30_count, label='>30')
    plt.bar(xs, no_count, bottom=lt30_count + gt30_count, label='NO')

    plt.xlabel('Clusters')
    plt.ylabel('Percentage of readmitted label')
    plt.legend()
    plt.show()

    print('Completeness:', metrics.completeness_score(truth, pred))
    print('Homogeneity:', metrics.homogeneity_score(truth, pred))
    print('Adjusted rand score:', metrics.adjusted_rand_score(truth, pred))
    print('Fowlkes-Mallows score:', metrics.fowlkes_mallows_score(truth, pred))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pred_file', type=str, default="kmeans_results_c6.csv")
    args = parser.parse_args()
    main(args)