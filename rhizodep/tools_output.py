# Import
import openalea.plantgl.all as pgl
from rhizodep.tools import plot_mtg
import matplotlib.pyplot as plt
import numpy as np


### Output parameters
vertices = [149]

state_extracts =  ['Nm',
                    'AA']

flow_extracts = ['import_Nm',
                'export_Nm',
                'xylem_Nm',
                'axial_advection_Nm_xylem',
                'axial_diffusion_Nm_xylem']


def plot_N(g, p, axs, span_slider):

    range_min, range_max = [0 for k in plot_properties], [0 for k in plot_properties]
    scene = pgl.Scene()
    for k in range(len(p)):
        # Computing plot ranges for the selected properties
        props = g.property(p[k])
        max_scale = g.max_scale()
        plot_range = [props[vid] for vid in g.vertices(scale=max_scale) if g.property("struct_mass")[vid] != 0]
        x_span = max(plot_range) - min(plot_range)
        range_min[k], range_max[k] = np.mean(plot_range) - span_slider*x_span/2, np.mean(plot_range) + span_slider*x_span/2
        # Creating a color mesh in a separate window for colormap's interpretation
        cm = plt.cm.get_cmap('jet')
        ax = axs[k]
        ax.clear()
        y,x = np.histogram(plot_range, 20)
        colors = [cm(((j - range_min[k]) / (range_max[k] - range_min[k]))) for j in x]
        ax.bar(x[:-1], y, color=colors, width=x[1]-x[0])

        scene += plot_mtg(g,
                         prop_cmap=p[k],
                         lognorm=False,  # to avoid issues with negative values
                         vmin=range_min[k],
                         vmax=range_max[k],
                         k=k)

    pgl.Viewer.display(scene)

    return range_min, range_max

def print_g(g, select, vertice):
    if vertice != 0:
        # extract MTG properties only once
        props = g.properties()
        extract = [props[k] for k in select]
        # print only selected segment
        print(vertice, end=' ')
        for k in range(len(extract)):
            print(' / ' + select[k] + ' :', end=' ')
            print("{:1.3e}".format(extract[k][vertice]), end=' ')
        print('')

    else:
        for k in select:
            print(k, getattr(g, k))

def plot_xr(dataset, vertice, select_state):
    fig = plt.figure()
    ax = fig.add_subplot()
    v_extract = dataset.sel(vid=vertice)
    for prop in select_state:
        getattr(v_extract, prop).plot.line(x='t', ax=ax)
    ax.legend(state_extracts[::-1])
    plt.show()