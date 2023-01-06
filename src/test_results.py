from metrics import *
from sklearn import preprocessing
import argparse
import pandas as pd


def main(args):
    data = pd.read_csv(args.data_pred_file)
    raw_data = data.iloc[:, 0:8].values
    min_max_scaler = preprocessing.MinMaxScaler()
    # normalize for k-means distance calculation
    minmax_data = min_max_scaler.fit_transform(raw_data)
    normalized_data = pd.DataFrame(minmax_data).apply(pd.to_numeric).to_numpy()
    
    pred = data.iloc[:, 8].values
    
    print("Analysing internal metrics...")
    print(f"Compactness: {compactness(normalized_data, pred)}")
    print(f"separation: {separation(normalized_data, pred)}")
    # print(f"dunn_validity_index_score: {dunn_validity_index_score(normalized_data, pred)}")
    print(f"SSE: {SSE(normalized_data, pred)}")
    # print(f"silhouette_score: {silhouette_score(normalized_data, pred)}")
    print(f"calinski_harabasz_score: {calinski_harabasz_score(normalized_data, pred)}")
    print(f"davies_bouldin_score: {davies_bouldin_score(normalized_data, pred)}")
    
    # print("Analysing cluster tendency...")
    # print(f"hopkins_statistics: {hopkins_statistics(normalized_data)}")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pred_file', type=str, default="kmeans_results_c12.csv")
    args = parser.parse_args()
    main(args)