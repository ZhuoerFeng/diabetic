from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize(X, X_pred, num_clusters):
    X_embedded = TSNE(n_components=2, learning_rate=20,
                    init='random', perplexity=3, verbose=1).fit_transform(X)
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
    for i in num_clusters:
        plt.scatter(
            X_embedded[X_pred == i, :],
            c=color_list[i]
        )
    plt.show()
