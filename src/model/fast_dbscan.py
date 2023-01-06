import numpy as np
from copy import deepcopy
from tqdm import tqdm

from scipy import spatial

from collections import namedtuple

DBSCANOutput = namedtuple(
    'DBSCANOutput', [
        'labels_',
        'cluster_centers_',
        'clusters'
    ]
)

NOISE = -1
UNLABELED = 0

class FastDBSCAN:
    def __init__(self, eps=0.1, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=10, p=None, n_jobs=None):
        self.eps = eps
        self.min_samples=min_samples
        self.metric = np.linalg.norm
        self.leaf_size = leaf_size
        self.p = p
        
        self.core_obj = list()
        self.data = None
        self.data_tag = list() # -1 represents noise
        self.clusters = dict()
        
    def setup_kdtree(self):
        print("Setting up KDTree...")
        print(type(self.data))
        print(self.data.shape)
        print(self.leaf_size)
        self.kdtree = spatial.KDTree(self.data, leafsize=self.leaf_size)
        print("KDTree set")
        
            
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
        return self.kdtree.query_ball_point(q, r=self.eps, workers=6)
    
    def fit(self, data):
        self.data = data
        self.setup_kdtree()
        core_obj = dict()
        self.data_tag = np.zeros(len(data), dtype=np.int)
        # save looking up time
        for i in range(len(self.data)):
            nei = self.get_neighbor_from_kdtree(i)
            if len(nei) >= self.min_samples:
                core_obj[i] = nei
        old_core_obj = core_obj.copy()
        
        class_num:int = 0
        unvisited = list(range(len(self.data)))
        while len(core_obj) > 0:
            print(f"Core obj len{len(core_obj)}")
            old_unvisited = list()
            old_unvisited.extend(unvisited)
            
            
            cur_idx = np.random.choice(list(core_obj.keys()))
            queue = [cur_idx]
            unvisited.remove(cur_idx)
            
            while len(queue) > 0:
                print(f"Queue len{len(queue)}")
                head = queue.pop()
                # head_nei = self.get_neighbor_from_kdtree(head)
                if head in old_core_obj.keys():
                    delta = [h_nei for h_nei in old_core_obj[head] if h_nei in unvisited]
                    queue.extend(delta)
                    unvisited = list((set(unvisited) - set(delta) ))
            class_num += 1
            self.clusters[class_num] = list(set(old_unvisited) - set(unvisited))
            for c in self.clusters[class_num]:
                self.data_tag[c] = class_num
                if c in core_obj.keys():
                    del core_obj[c]

        self.class_num = class_num
        return DBSCANOutput(
            labels_=self.data_tag,
            cluster_centers_=None,
            clusters=self.clusters
        ) 
