import openalea.plantgl.all as pgl
from rhizodep.tools import plot_mtg

def plot_N(g,
           p: str = 'influx_Nm'
           ):
    props = g.property(p)
    max_scale = g.max_scale()
    plot_range = [props[vid] for vid in g.vertices(scale=max_scale) if props[vid] != 0]

    scene = plot_mtg(g,
                     prop_cmap=p,
                     lognorm=False,  # to avoid issues with negative values
                     vmin=min(plot_range),
                     vmax=max(plot_range)
                     )
    pgl.Viewer.display(scene)


def print_g(g,
            select =
            ['influx_Nm',
              'loading_Nm',
              'soil_Nm',
              'Nm',
              'z1',
              'struct_mass'
              # 'C_hexose_root'
              # 'thermal_time_since_emergence'
              ],
            vertice: int = 0
            ):
    # extract MTG properties only once
    props = g.properties()
    extract = [props[k] for k in select]

    if vertice != 0:
        # print only selected segment
        print(vertice, end=' ')
        for k in range(len(extract)):
            print(select[k] + ' : ', end=' ')
            print(f"{extract[k][vertice]:4.15f}", end=' ')
        print('')

    else:
        max_scale = g.max_scale()
        for vid in g.vertices(scale=max_scale):
            # print for each segment selected properties in select
            print(vid, end=' ')
            for k in range(len(extract)):
                print(select[k] + ' : ', end=' ')
                print(f"{extract[k][vid]:4.15f}", end=' ')
            print('')
