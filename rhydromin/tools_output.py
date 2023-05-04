# Import
import openalea.plantgl.all as pgl
from rhydromin.tools import plot_mtg
import matplotlib.pyplot as plt
import numpy as np


### Output parameters
vertices = [149]

state_extracts = dict(
    # Next nitrogen properties
    Nm=dict(unit="mol N.g-1", value_example=float(1e-4), description="not provided"),
    AA=dict(unit="mol N.g-1", value_example=float(9e-4), description="not provided"),
    struct_protein=dict(unit="mol N.g-1", value_example=float(0), description="not provided"),
    storage_protein=dict(unit="mol N.g-1", value_example=float(0), description="not provided"),
    xylem_Nm=dict(unit="mol N.s-1", value_example=float(1e-4), description="not provided"),
    xylem_AA=dict(unit="mol N.s-1", value_example=float(1e-4), description="not provided"),
    phloem_AA=dict(unit="mol N.s-1", value_example=float(1e-4), description="not provided"),
    xylem_struct_mass=dict(unit="g", value_example=float(1e-3), description="not provided"),
    phloem_struct_mass=dict(unit="g", value_example=float(1e-3), description="not provided"),
    # Water model
    xylem_water=dict(unit="mol H2O", value_example=float(0), description="not provided"),
    # Topology model
    root_exchange_surface=dict(unit="m2", value_example=float(0), description="not provided"),
    stele_exchange_surface=dict(unit="m2", value_example=float(0), description="not provided"),
    phloem_exchange_surface=dict(unit="m2", value_example=float(0), description="not provided"),
    #apoplasmic_stele=dict(unit="adim", value_example=float(0.5), description="not provided"),
    xylem_volume=dict(unit="m3", value_example=float(0), description="not provided"),
    # Soil boundaries
    #soil_water_pressure=dict(unit="Pa", value_example=float(-0.1e6), description="not provided"),
    #soil_temperature=dict(unit="K", value_example=float(283.15), description="not provided"),
    #soil_Nm=dict(unit="mol N.m-3", value_example=float(0.5), description="not provided"),
    #soil_AA=dict(unit="mol AA.m-3", value_example=float(0), description="not provided")
)

flow_extracts = dict(
    import_Nm=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    export_Nm=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    export_AA=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_Nm_soil=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_Nm_xylem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_Nm_soil_xylem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_AA_soil=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_AA_phloem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_AA_soil_xylem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    AA_synthesis=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    struct_synthesis=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    storage_synthesis=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    AA_catabolism=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    storage_catabolism=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    cytokinin_synthesis=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    axial_advection_Nm_xylem = dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    axial_advection_AA_xylem = dict(unit="mol AA.s-1", value_example=float(0), description="not provided"),
    axial_diffusion_Nm_xylem = dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    axial_diffusion_AA_xylem = dict(unit="mol AA.s-1", value_example=float(0), description="not provided"),
    axial_diffusion_AA_phloem = dict(unit="mol AA.s-1", value_example=float(0), description="not provided"),
    # Water model
    radial_import_water=dict(unit="mol H2O.s-1", value_example=float(0), description="not provided"),
    axial_export_water_up=dict(unit="mol H2O.s-1", value_example=float(0), description="not provided"),
    axial_import_water_down=dict(unit="mol H2P.s-1", value_example=float(0), description="not provided"),
)

global_extracts = dict(
    total_Nm=dict(unit="mol", value_example="not provided",  description="not provided"),
    total_hexose=dict(unit="mol", value_example="not provided", description="not provided"),
    total_cytokinins=dict(unit="mol", value_example="not provided", description="not provided"),
    total_struct_mass=dict(unit="mol", value_example="not provided", description="not provided"),
    xylem_total_Nm=dict(unit="mol", value_example="not provided", description="not provided"),
    xylem_total_AA=dict(unit="mol", value_example="not provided", description="not provided"),
    phloem_total_AA=dict(unit="mol", value_example="not provided", description="not provided")
)


def plot_N(g, p, axs, span_slider):

    range_min, range_max = [0 for k in flow_extracts], [0 for k in flow_extracts]
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


def plot_xr(dataset, vertice=[], selection=[]):
    L = max(1, len(vertice))
    fig, ax = plt.subplots(L, 2)
    # If we plot global properties
    if len(vertice) == 0:
        ax = [ax]
        text_annot = [[]]
        std_dataset = (dataset - np.mean(dataset))/np.std(dataset)
        for prop in selection:
            getattr(dataset, prop).plot.line(x='t', ax=ax[0][0], label=prop)
            getattr(std_dataset, prop).plot.line(x='t', ax=ax[0][1], label=prop)
            text_annot[0] += [ax[0][0].text(0, 0, ""), ax[0][1].text(0, 0, "")]
    # If we plot local properties
    else:
        text_annot = [[] for k in range(len(vertice))]
        for k in range(len(vertice)):
            if len(vertice) > 1:
                modified_ax = ax[k]
            else:
                modified_ax = ax
            v_extract = dataset.sel(vid=vertice[k])
            std_v_extract = (v_extract - np.mean(v_extract))/np.std(v_extract)
            for prop in selection:
                getattr(v_extract, prop).plot.line(x='t', ax=modified_ax[0], label=prop)
                getattr(std_v_extract, prop).plot.line(x='t', ax=modified_ax[1], label=prop)
                text_annot[k] += [modified_ax[0].text(0, 0, ""), modified_ax[1].text(0, 0, "")]

    def hover(event):
        # for each row
        for axe in range(len(ax)):
            # for each column
            for norm in range(2):
                # if mouse event is in the ax
                if event.inaxes == ax[axe][norm]:
                    # At call remove all annotations to prevent overlap
                    for k in text_annot[axe]: k.set_visible(False)
                    # for all variables lines in the axe
                    for line in ax[axe][norm].get_lines():
                        # if the mouse pointer is on the line
                        cont, ind = line.contains(event)
                        if cont:
                            # get the position
                            posx, posy = [line.get_xdata()[ind['ind'][0]], line.get_ydata()[ind['ind'][0]]]
                            # get variable name
                            label = "{}:{}, {}".format(line.get_label(), posx, posy)
                            print(label)
                            # add text annotation to the axe and refresh
                            text_annot[axe] += [ax[axe][norm].text(x=posx, y=posy, s=label)]
                            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


# TODO : build coordinates after issue identification
# TODO : understand xylem conc explosion with in out flows.
