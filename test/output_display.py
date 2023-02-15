import openalea.plantgl.all as pgl
from rhizodep.tools import plot_mtg
import matplotlib.pyplot as plt
import numpy as np

def plot_N(g, range_min, range_max, p):

    if range_min[0] == 0 and range_max[0] == 0:
        plt.ion()
        fig, axs = plt.subplots(len(p), 1)
        for k in range(len(p)):
            # Computing once plot range for the selected property
            props = g.property(p[k])
            max_scale = g.max_scale()
            plot_range = [props[vid] for vid in g.vertices(scale=max_scale) if props[vid] != 0]
            range_min[k], range_max[k] = min(plot_range), max(plot_range)
            # Creating a color mesh in a separate window for colormap's interpretation
            x = np.array([range_min[k], range_max[k]])
            ax = axs[k]
            local_plot = ax.pcolormesh([x, x], cmap='jet', vmin=range_min[k], vmax=range_max[k])
            fig.colorbar(local_plot, ax=ax, location='top')
            ax.cla()
            ax.text(0, 1.4, p[k])
            ax.axis('off')

    scene = pgl.Scene()
    for k in range(len(p)):

        scene += plot_mtg(g,
                         prop_cmap=p[k],
                         lognorm=False,  # to avoid issues with negative values
                         vmin=range_min[k],
                         vmax=range_max[k],
                         k=k
                         )
    pgl.Viewer.display(scene)

    return range_min, range_max


def print_g(g, select, vertice):

    # extract MTG properties only once
    props = g.properties()
    extract = [props[k] for k in select]

    if vertice != 0:
        # print only selected segment
        print(vertice, end=' ')
        for k in range(len(extract)):
            print(' / ' + select[k] + ' :', end=' ')
            print("{:1.3e}".format(extract[k][vertice]), end=' ')
        print('')
        print(props['xylem_Nm'][1], props['xylem_AA'][1])

    else:
        max_scale = g.max_scale()
        for vid in g.vertices(scale=max_scale):
            # print for each segment selected properties in select
            print(vid, end=' ')
            for k in range(len(extract)):
                print(' / ' + select[k] + ' :', end=' ')
                print("{:e}".format(extract[k][vertice]), end=' ')
            print('')
        print(props['xylem_Nm'][1], props['xylem_AA'][1])
