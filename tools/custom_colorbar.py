import matplotlib.pyplot as plt
import matplotlib as mpl


def custom_colorbar(min=0, max=1, unit='Some Units'):
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.cm.get_cmap('jet')
    norm = mpl.colors.Normalize(vmin=min, vmax=max)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label(unit)
    plt.ion()
    fig.show()


if __name__ == '__main__':
    custom_colorbar()
