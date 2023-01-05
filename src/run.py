import pandas as pd
from sklearn import preprocessing
# from model import KMeans
from sklearn.cluster import KMeans
from utils import visualize, write_results_to_local_csv, radar

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
    n_clusters = 12
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10, verbose=1).fit(new_data.to_numpy()) # 558000
    visualize(new_data.to_numpy(), kmeans.labels_, n_clusters, filename="x_embed_10.npy")
    
    write_results_to_local_csv(
        column_name_list=column_names,
        data=useful_data,
        cluster_result=kmeans.labels_,
        filename="kmeans_results_c12.csv",
        overwrite=True,
        append=True
    )
    
    radar(kmeans.cluster_centers_, column_names)
    # k_means = KMeans(num_clusters=5, data=new_data.to_numpy()) # 199000
    # k_means.loop()

    
if __name__ == '__main__':
    main()