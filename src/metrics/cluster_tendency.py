import numpy as np
from sklearn.neighbors import NearestNeighbors

def hopkins_statistics(data, sample_ratio = 0.3):
    n_samples = int(data.shape[0] * sample_ratio)
    
    data_samples = data[np.random.choice(data.shape[0], n_samples, replace=False)]
    random_samples = np.random.uniform(np.min(data, axis=0), np.max(data, axis=0), (n_samples, data.shape[1]))
    nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(data)

    random_dists, _ = nearest_neighbors.kneighbors(random_samples, n_neighbors=2)
    data_dists, _ = nearest_neighbors.kneighbors(data_samples, n_neighbors=2)
    random_dists_sum = np.sum(random_dists[:, 0])
    data_dists_sum = np.sum(data_dists[:, 1])
    return random_dists_sum / (random_dists_sum + data_dists_sum)