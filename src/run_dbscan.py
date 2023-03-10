import pandas as pd
from sklearn import preprocessing
# from model import KMeans
from sklearn.cluster import DBSCAN
from utils import visualize, write_results_to_local_csv, radar
import numpy as np
import time

def main():
    data = pd.read_csv('data/diabetic_data_lht.csv')
    print(data.mean())

    # remove columns
    useful_data = data.iloc[:, 2:10].values
    column_names = data.columns[2:10].to_list()
    
    min_max_scaler = preprocessing.MinMaxScaler()
    # normalize for k-means distance calculation
    useful_data_scaled = min_max_scaler.fit_transform(useful_data)
    new_data = pd.DataFrame(useful_data_scaled).apply(pd.to_numeric)

    # new_data.apply(lambda s: pd.to_numeric(s, errors='raise').notnull().all())    

    print(new_data.mean())
    start = time.time()
    clustering = DBSCAN(eps=0.15, min_samples=5, algorithm="kd_tree", n_jobs=6).fit(new_data.to_numpy()) # 558000
    end = time.time()
    print(end - start)
    n_clusters = np.max(clustering.labels_)
    print(n_clusters)
    visualize(new_data.to_numpy(), clustering.labels_, n_clusters, filename="x_embed_10.npy")
    
    write_results_to_local_csv(
        column_name_list=column_names,
        data=useful_data,
        cluster_result=clustering.labels_,
        filename="dbscan_results_c12.csv",
        overwrite=True,
        append=False
    )
    
    radar(clustering.cluster_centers_, column_names)
    # k_means = KMeans(num_clusters=5, data=new_data.to_numpy()) # 199000
    # k_means.loop()

    
if __name__ == '__main__':
    main()