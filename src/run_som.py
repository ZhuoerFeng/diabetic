import pandas as pd
from sklearn import preprocessing
# from model import KMeans
from sklearn_som.som import SOM
from utils import visualize, write_results_to_local_csv, radar
from metric import culc_internal_metrics

def main():
    data = pd.read_csv('data/diabetic_data_lht.csv')
    print(data.mean())

    # remove columns
    useful_data = data.iloc[:, 2:-1].values
    column_names = data.columns[2:-1].to_list()
    
    min_max_scaler = preprocessing.MinMaxScaler()
    # normalize for k-means distance calculation
    useful_data_scaled = min_max_scaler.fit_transform(useful_data)
    new_data = pd.DataFrame(useful_data_scaled).apply(pd.to_numeric)

    # new_data.apply(lambda s: pd.to_numeric(s, errors='raise').notnull().all())    

    print(new_data.mean())
    for m in range(2, 6):
        for n in range(1, m):
            if m == 1 and n == 1:
                continue
            print('shape: ', m, '*', n)
            pred = SOM(m=m, n=n, dim=new_data.shape[1]).fit_predict(new_data.to_numpy())
            metrics = culc_internal_metrics(new_data.to_numpy(), pred)
            for k,v in metrics.items():
                print(k + ':', v)
            
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10, verbose=1).fit(new_data.to_numpy()) # 558000
    # visualize(new_data.to_numpy(), kmeans.labels_, n_clusters, filename="x_embed_10.npy")
    
    write_results_to_local_csv(
        column_name_list=column_names,
        data=useful_data,
        cluster_result=pred,
        filename="som_results_c6.csv",
        overwrite=True,
        append=False
    )
    
    radar(som.cluster_centers_, column_names)
    # k_means = KMeans(num_clusters=5, data=new_data.to_numpy()) # 199000
    # k_means.loop()

    
if __name__ == '__main__':
    main()