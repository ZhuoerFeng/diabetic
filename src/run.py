import pandas as pd
from sklearn import preprocessing
# from model import KMeans
from sklearn.cluster import KMeans
from utils import visualize

def main():
    data = pd.read_csv('data/diabetic_data_lht.csv')
    print(data.mean())

    # remove columns
    useful_data = data.iloc[:, 2:10].values
    min_max_scaler = preprocessing.MinMaxScaler()
    # normalize for k-means distance calculation
    useful_data_scaled = min_max_scaler.fit_transform(useful_data)
    new_data = pd.DataFrame(useful_data_scaled).apply(pd.to_numeric)

    # new_data.apply(lambda s: pd.to_numeric(s, errors='raise').notnull().all())    

    print(new_data.mean())
    n_clusters = 12
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10, verbose=1).fit(new_data.to_numpy()) # 558000
    visualize(new_data.to_numpy(), kmeans.labels_, n_clusters, filename="x_embed_10.npy")
    # k_means = KMeans(num_clusters=5, data=new_data.to_numpy()) # 199000
    # k_means.loop()

    
if __name__ == '__main__':
    main()