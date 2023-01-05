from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize(X, X_pred, num_clusters, filename='x_embedded.npy', recalc=False):
    if os.path.exists(filename) and not recalc:
        X_embedded = np.load(filename)
    else:
        X_embedded = TSNE(n_components=2, init='pca', random_state=42, verbose=1).fit_transform(X)
        print(X_embedded.shape)
        print('finished tnse')
        np.save(filename, X_embedded)
    color_list = [
        "#fff7bc",
        "#fec44f",
        "#d95f0e",
        "#f7fcb9",
        "#addd8e",
        "#31a354",
        "#e0ecf4",
        "#9ebcda",
        "#8856a7",
        "#edf8b1",
        "#7fcdbb",
        "#2c7fb8",
    ]
    for i in range(num_clusters):
        plt.scatter(
            X_embedded[X_pred == i, 0],
            X_embedded[X_pred == i, 1],
            c=color_list[i],
            label=f"$cluster {i}$"
        )
    plt.legend()
    plt.show()
