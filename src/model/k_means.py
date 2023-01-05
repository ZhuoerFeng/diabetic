import numpy as np
from copy import deepcopy
from collections import namedtuple

KMeansOutput = namedtuple(
    'KMeansOutput', [
        'labels_',
        'cluster_centers_',
        'inertia'
    ]
)

class KMeans:
    def __init__(self, n_clusters: int=5, random_state:int = 0, max_iter: int=40, n_init: int=10, distance_fn = None, verbose:int = 0):
        # np.random.seed(random_state)
        self.n_clusters = n_clusters
        if distance_fn is None:
            self.distance_fn = np.linalg.norm
        else:
            self.distance_fn = distance_fn
        self.tolerate = 1.0
        self.max_iter = max_iter
        self.n_init = n_init
    
    def init_data_attr(self, data):
        self.data = data
        print(self.data.shape)
        self.data_tag = np.zeros(data.shape[0], dtype=np.int)
        datadim = data.shape[1]
        self.centroids = np.random.rand(self.n_clusters, datadim)
        self.best_tag = None
        self.best_centroid = None
        self.best_inertia = None
        
    def init_iteration(self):
        self.data_tag = np.zeros(self.data.shape[0], dtype=np.int)
        self.centroids = np.random.rand(self.n_clusters, self.data.shape[1])
        
    def kmeans_plus_plus_init(self):
        self.data_tag = np.zeros(self.data.shape[0], dtype=np.int)
        start = np.random.choice(np.arange(self.data.shape[0]))
        chosen_list = [start]
        self.centroids[0] = self.data[start]
        for idx in range(self.n_clusters - 1):
            c_idx = idx + 1 # current centroid being chosen
            prob = np.zeros(self.data.shape[0], dtype=np.float)
            # calc distances
            for n_idx in range(self.data.shape[0]):
                if n_idx in chosen_list:
                    continue
                else:
                    prob[n_idx] = np.max([self.distance_fn(self.data[n_idx] - self.centroids[c]) for c in range(c_idx) ]) # calc longest dist from chosen centroids
            prob = prob * prob / np.sum(prob * prob)
            new_choice = np.random.choice(np.arange(self.data.shape[0]), p=prob)
            chosen_list.append(new_choice)
            self.centroids[c_idx] = self.data[new_choice]


    def update_datatag(self):
        for idx, data in enumerate(self.data):
            distances = [self.distance_fn(data - c, ord=2) for c in self.centroids]
            # print(np.argmin(distances))
            self.data_tag[idx] = np.argmin(distances)
            
    def update_cluster_centroid(self):
        for idx in range(self.n_clusters):
            self.centroids[idx] = np.mean(
                self.data[self.data_tag == idx, :],
                axis=0
            )
            
    def calc_mean_dist(self):
        dist = list()
        for idx, data in enumerate(self.data):
            dist.append(self.distance_fn(data - self.centroids[self.data_tag[idx], :], ord=2) ** 2)
        return np.sum(dist, axis=0)


    def single_iteration(self):
        prv_inertia = -1
        # self.init_iteration()
        self.kmeans_plus_plus_init()
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
        for i in range(self.n_init):
            inertia = self.single_iteration()
            print(f"Epoch {i}, inertia = {inertia}")
            if self.best_inertia is None or self.best_inertia > inertia:
                best_epoch = i
                self.best_inertia = inertia
                self.best_tag = deepcopy(self.data_tag)
                self.best_centroid = deepcopy(self.best_centroid)
                print("Update best inertia")
        print(f"Found best inertia {self.best_inertia} in loop epoch {best_epoch}")
        return KMeansOutput(
            labels_=self.best_tag,
            cluster_centers_=self.best_centroid,
            inertia=self.best_inertia
        )
    
    def fit(self, data):
        self.init_data_attr(data)
        return self.loop()