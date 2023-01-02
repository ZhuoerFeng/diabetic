from sklearn import datasets
import numpy as np
from model import DBSCAN
import matplotlib.pyplot as plt

def main():
    X,Y1 = datasets.make_circles(n_samples = 1500, factor = .4, noise = .07)
    print(X.shape)
    print(Y1.shape)
    model = DBSCAN()
    tag, class_num = model.fit(X)

    print(class_num)
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