import numpy as np
from copy import deepcopy

class KMeans:
    def __init__(self, num_clusters: int=5, max_iter: int=40, n_times: int=10, data = None, distance_fn = None):
        self.num_clusters = num_clusters
        if distance_fn is None:
            self.distance_fn = np.linalg.norm
        else:
            self.distance_fn = distance_fn
        self.data = data
        print(self.data.shape)
        self.data_tag = np.zeros(data.shape[0], dtype=np.int)
        datadim = data.shape[1]
        self.centroids = np.random.rand(num_clusters, datadim)
        self.max_iter = max_iter
        self.n_times = n_times
        self.tolerate = 3e-5
        
        self.best_tag = None
        self.best_centroid = None
        self.best_inertia = None
        
    def init_iteration(self):
        self.data_tag = np.zeros(self.data.shape[0], dtype=np.int)
        self.centroids = np.random.rand(self.num_clusters, self.data.shape[1])

    def update_datatag(self):
        for idx, data in enumerate(self.data):
            distances = [self.distance_fn(data - c, ord=2) for c in self.centroids]
            # print(np.argmin(distances))
            self.data_tag[idx] = np.argmin(distances)
            
    def update_cluster_centroid(self):
        for idx in range(self.num_clusters):
            self.centroids[idx] = np.mean(
                self.data[self.data_tag == idx, :],
                axis=0
            )
            
    def calc_mean_dist(self):
        dist = list()
        for idx, data in enumerate(self.data):
            dist.append(self.distance_fn(data - self.centroids[self.data_tag[idx], :], ord=2))
        return np.sum(dist, axis=0)


    def single_iteration(self):
        prv_inertia = -1
        self.init_iteration()
        for i in range(self.max_iter):
            self.update_datatag()
            self.update_cluster_centroid()
            inertia = self.calc_mean_dist()
            print(f"Iterate {i}: inertia = {inertia}")
            if np.abs(prv_inertia - inertia) < self.tolerate:
                prv_inertia = inertia
                print("Within tolerate, break")
                break
            prv_inertia = inertia
        return inertia
    
    def loop(self):
        best_epoch = 0
        for i in range(self.n_times):
            inertia = self.single_iteration()
            print(f"Epoch {i}, inertia = {inertia}")
            if self.best_inertia is None or self.best_inertia > inertia:
                best_epoch = i
                self.best_inertia = inertia
                self.best_tag = deepcopy(self.data_tag)
                self.best_centroid = deepcopy(self.best_centroid)
                print("Update best inertia")
        print(f"Found best inertia {self.best_inertia} in loop epoch {best_epoch}")
        return self.best_inertia, self.best_centroid, self.best_tag