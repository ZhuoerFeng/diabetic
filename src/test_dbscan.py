from sklearn import datasets
import numpy as np
from model import DBSCAN, FastDBSCAN
import matplotlib.pyplot as plt

def main():
    X,Y1 = datasets.make_circles(n_samples = 75000, factor = .4, noise = .07)
    print(X.shape)
    print(Y1.shape)
    model = FastDBSCAN(eps=0.10,  min_samples=100)
    tag, _, _ = model.fit(X)

    print(np.max(tag))
    plt.scatter(
        X[tag==1, 0],
        X[tag==1, 1],
        c="red"
    )
    plt.scatter(
        X[tag==2, 0],
        X[tag==2, 1],
        c="blue"
    )
    plt.scatter(
        X[tag==-1, 0],
        X[tag==-1, 1],
        c="grey"
    )
    plt.show()
    
    # plt.scatter(
    #     X[Y1==0, 0],
    #     X[Y1==0, 1],
    #     c="red"
    # )
    # plt.scatter(
    #     X[Y1==1, 0],
    #     X[Y1==1, 1],
    #     c="blue"
    # )
    # plt.show()
    
if __name__ == '__main__':
    main()