from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import pandas as pd

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


from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

def write_results_to_local_csv(column_name_list: list, data: np.ndarray, cluster_result, filename: str, overwrite=False, append=True):
    if not overwrite and os.path.exists(filename):
        raise Exception(f"File {filename} already exists")
    if not append:
        f = open(filename, 'w')
        writer = csv.writer(f, delimiter=',')
        writer.writerow(column_name_list + ['cluster_result'])
        for idx, line in enumerate(data):
            writer.writerow(
                line.tolist() + [cluster_result[idx]]
            )
    else:
        prev_res = pd.read_csv(filename)
        new_col_name = f"added_column_{len(prev_res.columns)}"
        prev_res[new_col_name] = cluster_result
        prev_res.to_csv(filename, index=False)
        
        
def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    def draw_poly_patch(self):
        # rotate theta such that the first axis is at the top
        verts = unit_poly_verts(theta + np.pi / 2)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def __init__(self, *args, **kwargs):
            super(RadarAxes, self).__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta + np.pi / 2)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts


def radar(centroids, labels):
    N = 8
    theta = radar_factory(N, frame='polygon')
    spoke_labels = labels
    fig, axes = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                             subplot_kw=dict(projection='radar'))  ###画布大小
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.95, bottom=0.05)

    colors = ['b', 'r', 'g', 'm', 'y','c','k','w']
    # for ax in axes.flatten():
    axes.set_rgrids([0.2, 0.4, 0.6, 0.8])
    axes.set_title('radar', weight='bold', size='medium', position=(0.5, 1.1),
                    horizontalalignment='center', verticalalignment='center')
    for d, color in zip(centroids, colors):
        axes.plot(theta, d[0:8], color=color)
        axes.fill(theta, d[0:8], facecolor=color, alpha=0.25)
    axes.set_varlabels(spoke_labels)
    # ax = axes[0, 0]
    labels = ('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5')
    legend = axes.legend(labels, loc=(0.9, .95),
                       labelspacing=0.1, fontsize='small')

    plt.savefig('randar.png')

