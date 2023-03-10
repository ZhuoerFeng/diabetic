import numpy as np
from copy import deepcopy
from tqdm import tqdm

from scipy import spatial

from collections import namedtuple

DBSCANOutput = namedtuple(
    'DBSCANOutput', [
        'labels_',
        'cluster_centers_',
        'inertia'
    ]
)

NOISE = -1
UNLABELED = 0

class DBSCAN:
    def __init__(self, eps=0.1, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=10, p=None, n_jobs=None):
        self.eps = eps
        self.min_samples=min_samples
        self.metric = np.linalg.norm
        self.leaf_size = leaf_size
        self.p = p
        
        self.core_obj = list()
        self.data = None
        self.data_tag = list() # -1 represents noise
        
    def setup_kdtree(self):
        print("Setting up KDTree...")
        print(type(self.data))
        print(self.data.shape)
        print(self.leaf_size)
        self.kdtree = spatial.KDTree(self.data, leafsize=self.leaf_size)
        print("KDTree set")
        
    def init_fit(self):
        self.core_obj = list()
            
    def get_neighbor(self, point_idx):
        neighbor = list()
        p = self.data[point_idx]
        for i, data in enumerate(self.data):
            if self.metric(p - data, ord=2) < self.eps:
                if i != point_idx:
                    neighbor.append(i)
        return neighbor
    
    def get_neighbor_from_kdtree(self, point_idx):
        q = self.data[point_idx]
        return self.kdtree.query_ball_point(q, r=self.eps)
    
    def fit(self, data):
        self.data = data
        self.setup_kdtree()
        self.data_tag = np.zeros(len(data), dtype=np.int)
        
        class_num:int = 0
        for i in tqdm(range(len(data))):
            if self.data_tag[i] != 0:
                continue
            # neighbor = self.get_neighbor(i)
            neighbor = self.get_neighbor_from_kdtree(i)
            if len(neighbor) < self.min_samples:
                self.data_tag[i] = NOISE
                continue
            class_num = class_num + 1
            self.data_tag[i] = class_num
            
            waiting_list = neighbor
            while len(waiting_list) > 0:
                check_id = waiting_list.pop()
                if self.data_tag[check_id] == NOISE:
                    self.data_tag[check_id] = class_num
                if self.data_tag[check_id] != 0:
                    continue
                self.data_tag[check_id] = class_num
                # new_neighbor = self.get_neighbor(check_id)
                new_neighbor = self.get_neighbor_from_kdtree(check_id)
                if len(new_neighbor) >= self.min_samples: # ?????????
                    for n in new_neighbor:
                        if n not in waiting_list:
                            waiting_list.append(n)

        self.class_num = class_num
        return DBSCANOutput(
            labels_=self.data_tag,
            cluster_centers_=None,
            inertia=None
        ) 
