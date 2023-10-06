'''
Special requirements instructions:
__________________________________
conda install -c conda-forge xarray dask netCDF4 bottleneck
install matplotlib before tensorflow for compatibility issues
pandas, pickle
python -m pip install scikit-learn
conda install -c conda-forge umap-learn
python -m pip install tensorflow==2.12.0
conda install -c conda-forge hdbscan
'''

'''IMPORTS'''
# Data processing packages
import pandas as pd
import xarray as xr
import xbatcher as xb
import numpy as np
# Visual packages
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib.path import Path
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Tensor management packages
from tensorflow import keras, reshape
from keras.layers import MaxPool2D, Conv2D, Conv2DTranspose, ReLU, UpSampling2D, Activation, Flatten, Dense, Reshape, Input
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Projection
import umap
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import f_oneway

'''FUNCTIONS'''


class Preprocessing:
    def __init__(self, central_dataset, coordinates={}, type='csv', variables={}, window=24, stride=12):
        self.unormalized_ds = central_dataset[list(variables.keys())]
        del central_dataset
        self.normalized_ds = self.normalization(self.unormalized_ds).fillna(0)

        # stacking to put every sliced window on the same learning slope then
        self.normalized_ds = self.normalized_ds.stack(stk=[dim for dim in self.normalized_ds.dims if dim != "t"])
        n_windows = int(1 + ((max(self.normalized_ds.coords["t"].values)-window + 1)/stride))

        self.labels, self.t_windows = [], []
        for coord in self.normalized_ds.coords["stk"].values:
            self.t_windows += [k*stride for k in range(n_windows)]
            self.labels += [coord]*n_windows

        bgen = xb.BatchGenerator(self.normalized_ds.to_array().transpose("stk", "variable", "t"), input_dims={'stk': 1, 't': window}, input_overlap={"t": window-stride}, preload_batch=True)
        depth = 12
        self.stacked_da = np.concatenate([[reshape(batch, shape=(1, window, depth))] for batch in bgen])

    def normalization(self, dataset):
        """
        Standard normalization technique
        """
        return (dataset - dataset.min(dim="t")) / (dataset.max(dim="t") - dataset.min(dim="t"))


class DCAE:
    @staticmethod
    def build(height=1, width=60, depth=6, filters=((64, 10, 2), (32, 5, 2), (12, 5, 3)), latentDim=60):
        # initialize the input shape to be "channels last" along with
        # the channels dimension itself
        # channels dimension itself
        inputShape = (height, width, depth)

        # define the input to the encoder
        inputs = Input(shape=inputShape)
        x = inputs
        # loop over the number of filters
        for f, s, p in filters:
            # apply a CONV => RELU => BN operation
            # Limites Relu non dérivable donc approximation limitée pour faible gradient, voir leaky relu
            x = Conv2D(f, (1, s), strides=1, padding="same")(x)
            x = ReLU()(x)
            x = MaxPool2D(pool_size=(1, p))(x)
        # flatten the network and then construct our latent vector
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim, activation='linear')(x)
        # build the encoder model
        encoder = Model(inputs, latent, name="encoder")

        # start building the decoder model which will accept the
        # output of the encoder as its inputs
        latentInputs = Input(shape=(latentDim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
        # loop over our number of filters again, but this time in
        # reverse order
        for f, s, p in filters[::-1]:
            # apply a CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f, (1, s), strides=1, padding="same")(x)
            x = ReLU()(x)
            x = UpSampling2D(size=(1, p))(x)

        # apply a single CONV_TRANSPOSE layer used to recover the
        # original depth of the image
        x = Conv2DTranspose(depth, (1, filters[0][0]), padding="same")(x)
        outputs = Activation("linear")(x)

        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")
        # our autoencoder is the encoder + decoder
        autoencoder = Model(inputs, decoder(encoder(inputs)),
                            name="autoencoder")

        # return a 3-tuple of the encoder, decoder, and autoencoder
        return (encoder, decoder, autoencoder)

    @staticmethod
    def train(stacked_dataset, autoencoder, test_prop=0.2, epochs=25, batch_size=100, plotting=False):
        trainX, testX = train_test_split(stacked_dataset, test_size=test_prop)
        opt = Adam(learning_rate=1e-3)
        autoencoder.compile(loss="mse", optimizer=opt)
        # train the convolutional autoencoder
        H = autoencoder.fit(
            trainX, trainX,
            validation_data=(testX, testX),
            epochs=epochs,
            batch_size=batch_size)

        if plotting:
            # construct a plot that plots and saves the training history
            N = np.arange(0, epochs)
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, H.history["loss"], label="train_loss")
            plt.plot(N, H.history["val_loss"], label="val_loss")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            plt.show()

        return autoencoder


class MainMenu:
    def __init__(self, windows_ND_projection, latent_windows, sliced_windows, original_unorm_dataframe, original_dataframe, coordinates, clusters, index_2D=[], index_ND=[], window=60, plot=False):
        self.windows_ND_projection = windows_ND_projection
        self.latent_windows = latent_windows
        self.sliced_windows = sliced_windows
        self.original_unorm_dataframe = original_unorm_dataframe
        self.original_dataframe = original_dataframe
        self.coordinates = coordinates
        self.window = window
        self.index_2D = index_2D
        self.index_ND = index_ND
        self.vid_numbers = np.unique(self.coordinates.index.get_level_values("vid"))
        self.segmented_coords = [
            np.concatenate(self.coordinates[[index[0] == layer for index in self.coordinates.index]].values) for layer
            in self.vid_numbers]

        if plot:
            # Color points by layer category
            self.layer_colors = []
            M = self.coordinates.values.max()
            L = len(self.latent_windows)
            for layer in range(L):
                self.layer_colors += [(k/M, 0, 1 - (k/M)) for k in self.segmented_coords[layer][:-(self.window-1)]]

            self.fig1 = plt.figure(figsize=(10, 14))
            self.ax = self.fig1.add_subplot(projection='3d')

            # Aesthetic properties
            self.fig1.tight_layout()
            self.title = self.ax.text2D(0.5, 0.9, s="Latent windows Umap 3d projection (color = spatial coordinates)",
                                        ha="center", transform=self.ax.transAxes, fontsize=12)
            self.ax.xaxis.set_ticklabels([])
            self.ax.yaxis.set_ticklabels([])
            self.ax.zaxis.set_ticklabels([])
            self.ax.grid(False)

        # Tk init
        self.root = tk.Tk()
        self.root.title('ND UMAP projection of extracted windows')
        #self.root.state("zoomed")
        if plot:
            # Figure
            tk_plot = FigureCanvasTkAgg(self.fig1, self.root)
            tk_plot.get_tk_widget().grid(row=0, column=0, rowspan=4)
            # Buttons widget
            toggle_button = tk.Button(self.root, text='Select/Rotate', command=self.toggle_selector)
            toggle_button.grid(row=3, column=1)
            del_button = tk.Button(self.root, text='Del Annot', command=self.remove_annotations)
            del_button.grid(row=3, column=2)
            color_button = tk.Button(self.root, text='t/coord color', command=self.toggle_color)
            color_button.grid(row=3, column=4)
            self.ax.disable_mouse_rotation()
            self.select = False
            self.time_color = False

        # Listbox widget
        self.lb = tk.Listbox(self.root)
        for k in range(len(self.vid_numbers)):
            self.lb.insert(k, str(self.vid_numbers[k]))
        self.lb.grid(row=1, column=2, sticky='N')
        # Buttons widget

        plot_button = tk.Button(self.root, text='Topo slice', command=self.flat_plot_instance)
        plot_button.grid(row=2, column=2, sticky='N')

        info_button = tk.Button(self.root, text='Clusters info', command=self.cluster_info)
        info_button.grid(row=3, column=3)

        # Label widget
        label = tk.Label(self.root, text='Organ ID :')
        label.grid(row=0, column=2, sticky='S')

        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=10)
        self.root.rowconfigure(3, weight=1)
        # To ensure regular column spacing after graph
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.columnconfigure(3, weight=1)
        self.root.columnconfigure(4, weight=1)

        self.label = []
        self.clusters = clusters


    def select_callback_UMAP(self, verts):
        path = Path(verts)
        selected = np.nonzero(path.contains_points(self.pts.get_offsets()))[0]
        if len(selected) > 0:
            self.clusters += [selected]
            self.update_colors()

    def toggle_selector(self):
        if self.select:
            print('Selector activated')
            self.RS.set_active(True)
            self.ax.disable_mouse_rotation()
            self.select = False
        else:
            print('Rotation activated')
            self.RS.set_active(False)
            self.ax.mouse_init()
            self.select = True

    def toggle_color(self):
        if self.time_color:
            self.layer_colors = []
            M = self.coordinates.values.max()
            L = len(self.latent_windows)
            for layer in range(L):
                self.layer_colors += [(k / M, 0, 1 - (k / M)) for k in
                                      self.segmented_coords[layer][:-(self.window - 1)]]

            self.update_colors()

            # Colorbar update
            self.sm.set_clim(vmin=self.coordinates.values.min(), vmax=self.coordinates.values.max())
            self.cb.set_ticks(np.arange(self.coordinates.values.min(), self.coordinates.values.max(), 1e-2))
            self.cb.set_label("Topology coordinates")

            self.title.set_text("Latent windows Umap 3d projection (color = spatial coordinates)")
            self.time_color = False
        else:
            times = []
            tp = [[i for i in range(len(k))] for k in self.latent_windows]
            for k in tp:
                times += k
            m = max(times)
            self.layer_colors = [(time/m, 0, 1-time/m) for time in times]
            self.update_colors()

            # To compute a good colorbar, an ordered list of unique times is needed
            unique_times = sorted(np.unique(times))
            idx = np.round(np.linspace(0, len(unique_times) - 1, 7)).astype(int)
            # Update colorbar's colormap
            self.sm.set_clim(vmin=min(idx), vmax=max(idx))
            self.cb.set_ticks(idx)
            self.cb.set_ticklabels([unique_times[k] for k in idx])
            self.cb.set_label("Time coordinates")

            self.title.set_text("Latent windows Umap 3d projection (color = time coordinates)")
            self.time_color = True

    def update_colors(self, which=-1):
        if len(self.clusters) > 0:
            color = (0, 0.1 + 0.9*np.random.random(), 0, np.random.random())
            T = list(self.clusters[which])
            if len(T) > 0:
                cluster_points = self.windows_ND_projection[T]

                self.label += [self.ax.text(np.mean(cluster_points[:,0]), np.mean(cluster_points[:,1]), np.mean(cluster_points[:,2]),
                                            s=str(len(self.label)), fontweight='bold')]
                for k in T:
                    self.layer_colors[k] = color
        #self.pts.set_color(self.layer_colors)
        plt.draw()

    def remove_annotations(self):

        self.clusters = []
        for k in self.label:
            k.remove()
        self.label = []
        self.layer_colors = []
        M = self.coordinates.values.max()
        L = len(self.latent_windows)
        for layer in range(L):
            self.layer_colors += [(k / M, 0, 1 - (k / M)) for k in self.segmented_coords[layer][:-(self.window - 1)]]
        self.pts.set_color(self.layer_colors)
        self.toggle_color()
        self.toggle_color()


    def svm_selection(self, properties=[""]):
        classes = []
        selected_groups = []
        conc_latent_windows = np.concatenate(self.latent_windows)
        if len(self.clusters) == 0:
            print("[Error] : No selection")
        elif len(self.clusters) == 1:
            self.clusters += [[k for k in range(len(self.windows_ND_projection)) if k not in self.clusters[0]]]
            self.update_colors()
        # Retrieving latent windows corresponding to selected groups
        for k in range(len(self.clusters)):
            classes += [k for j in range(len(self.clusters[k]))]
            selected_groups += [list(i) for i in conc_latent_windows[self.clusters[k]]]
        if len(self.clusters) > 1:
            # splitting data for svm training and test
            x_train, x_test, y_train, y_test = train_test_split(selected_groups, classes, test_size=0.2)
            clf = SVC(kernel='linear', C=100)
            clf.fit(selected_groups, classes)
            result = clf.predict(x_test)
            # Evaluating the accuracy of the model using the sklearn functions
            accuracy = accuracy_score(y_test, result) * 100
            confusion_mat = confusion_matrix(y_test, result)

            # Printing the results
            print("Accuracy for SVM is:", accuracy)
            print("Confusion Matrix")
            print(confusion_mat)
        else:
            print("Only one class")

        # Check individual variables contributions to differences between clusters
        # for each labelled cluster
        nb_props = len(properties)
        # grouping separated variable windows to use indexes selected by clusters
        conc_sliced_windows = np.concatenate(self.sliced_windows)
        clusters_windows = [conc_sliced_windows[k] for k in self.clusters]
        # For each of the clusters, for each variable, we compute the mean of values for t0, t1, ...
        # Index 0 is used for windows because that's how the DCAE input shape was designed (1 x window x nb_props)
        curve_sets_clusters = [[[[win[0][t][v] for win in clst] for t in range(self.window)] for v in range(nb_props)] for clst in clusters_windows]
        # (unit = sum(flux_unit*time_step) = quantity per time_step hours)

        # Matrix to present main responsible for divergence between clusters through the Area Under the Curve (AUC)
        aucs = [[{} for k in range(len(self.clusters))] for i in range(len(self.clusters))]
        # for each row
        for k in range(len(self.clusters)):
            # for each element after diagonal in row
            for l in range(k+1, len(self.clusters)):
                # For a given variable, get the differences of means at each time-step
                # Then sum this to compute Area Under the curve for each variable and add it in the cross comparison matrix
                # for each variable, label the sum to a name (dict key) for readability
                for v in range(nb_props):
                    aucs[k][l][properties[v]] = np.sum([(np.mean(curve_sets_clusters[k][v][t]) - np.mean(curve_sets_clusters[l][v][t])) *
                                                        (1-f_oneway(curve_sets_clusters[k][v][t], curve_sets_clusters[l][v][t]).pvalue) for t in range(self.window)
                                                        if not np.isnan(f_oneway(curve_sets_clusters[k][v][t], curve_sets_clusters[l][v][t]).pvalue)])

        return aucs

    def cluster_info(self):
        print("[INFO] building cluster visualisation")
        if len(self.clusters) == 0:
            print("[Error] : no cluster selected")
            return
        properties = self.original_dataframe[0].columns
        aucs = self.svm_selection(properties=properties)
        fig3 = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, len(self.clusters), height_ratios=[1, 2], figure=fig3)

        ax30 = [fig3.add_subplot(gs[0, k]) for k in range(len(self.clusters))]
        ax31 = fig3.add_subplot(gs[1, :])

        fig3.text(0.01, 0.95, "Space-Time repartition", fontweight='bold')
        fig3.text(0.01, 0.50, "window AUC between clusters", fontweight='bold')

        heatmap = []
        pair_labels = []
        # for each cluster combination
        per_organ_count = [[0 for i in range(len(self.latent_windows))] for k in range(len(self.clusters))]
        for k in range(len(self.clusters)):
            times = []
            coords = []
            for i in range(len(self.index_ND)):
                per_organ_count[k][i] = sum([j in self.index_ND[i] for j in self.clusters[k]])
                per_class_times = [self.index_ND[i].index(j) for j in self.clusters[k] if j in self.index_ND[i]]
                # for this organ, if times have been selected, the related coordinates are to be displayed
                per_class_coords = [self.segmented_coords[i][t] for t in per_class_times]
                times += per_class_times
                coords += per_class_coords

            maxs_index = []
            max_organs = max(per_organ_count[k])
            for j in range(len(per_organ_count[k])):
                if per_organ_count[k][j] == max_organs:
                    maxs_index += [self.vid_numbers[j]]

            ax30[k].set_title("C" + str(k) + " : " + str(int(len(self.clusters[k])/1000)) + "k / " + str(maxs_index[1:4])[1:-1])
            ax30[k].hist2d(times, coords, bins=20, cmap="Purples")

            for i in range(k+1, len(self.clusters)):
                heatmap += [list(aucs[k][i].values())]
            pair_labels += ["{}v{}".format(k, i) for i in range(k + 1, len(self.clusters))]

        hm = ax31.imshow(np.transpose(heatmap), cmap="PiYG", aspect="auto")
        fig3.colorbar(hm, orientation='horizontal', location='top')

        ax31.set_xticks(np.arange(len(pair_labels)), labels=pair_labels)
        ax31.set_yticks(np.arange(len(properties)), labels=properties)

        # Loop over data dimensions and create text annotations.
        for i in range(len(pair_labels)):
            for j in range(len(properties)):
                ax31.text(i, j, int(round(heatmap[i][j], 0)), ha="center", va="center", color="w",
                               fontsize=10, fontweight='bold')
        fig3.show()

    def build_app(self, plot=False):
        # time is an axis as others, rather, default color corresponding coordinates on structure
        # For a given cluster, it wil enable user to select 2D plots of interest for targeted layers, WITH corresponding clusters highlighted (and refreshed)

        if plot:
            self.pts = self.ax.scatter(
                self.windows_ND_projection[:, 0],
                self.windows_ND_projection[:, 1],
                self.windows_ND_projection[:, 2],
                c=self.layer_colors,
                s=2
            )


            # ticks are coordinates between 0 and 1, but then you name them different with set_ticklabel()
            my_cmap = LinearSegmentedColormap.from_list("custom", [(0,0,1), (1,0,0)], N=50)
            self.sm = plt.cm.ScalarMappable(cmap=my_cmap)
            self.sm.set_clim(vmin=self.coordinates.values.min(), vmax=self.coordinates.values.max())
            self.cb = self.fig1.colorbar(self.sm, ticks=np.arange(self.coordinates.values.min(), self.coordinates.values.max(), 1e-2), label="Topology coordinates", shrink=0.5)

            # to account for clusters at window opening
            for k in range(len(self.clusters)):
                self.update_colors(which=k)

            self.RS = LassoSelector(self.ax, self.select_callback_UMAP, useblit=True, button=[1, 3])

        self.cluster_info()
        self.root.mainloop()

    def flat_plot_instance(self):
        layer = self.lb.curselection()
        if len(layer) > 0:
            layer = self.vid_numbers[layer[0]]
            # retrieve index from function(event)
            lp = LinkPlotting(layer=layer, original_unorm_dataframe=self.original_unorm_dataframe, original_dataframe=self.original_dataframe,
                                     clusters=self.clusters, index_2D=self.index_2D, index_ND=self.index_ND, window=self.window)


# Get back to classical instance
class LinkPlotting:
    def __init__(self, layer, original_unorm_dataframe, original_dataframe, clusters=[], index_2D=[], index_ND=[], window=60):
        self.layer = layer
        self.original_unorm_dataframe = original_unorm_dataframe[layer]
        self.original_dataframe = original_dataframe[layer]
        self.window = window
        self.index_2D = index_2D
        self.index_ND = index_ND

        self.axvspan_list = []
        self.clusters, self.clusters_label = self.selection_Nd_to_2d(cluster_Nd=clusters)

        self.fig2 = plt.figure(figsize=(12, 10))
        self.fig2.suptitle('Extract for topological coordinate "{}"'.format(layer), fontsize=12)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        self.axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]
        self.axs[1].axis("off")

        self.build_plot()

    def update_colors(self, which="last"):
        if which == "last":
            cl = [list(self.clusters[-1])]
        elif which == "all":
            cl = [list(k) for k in self.clusters]
        for T in cl:
            color = np.random.rand(3, )
            # 1D time-series selection
            xmin = T[0]
            for k in range(len(T)):
                if k < len(T)-1:
                    if T[k] + self.window not in T and T[k+1] > T[k] + self.window:
                        self.axvspan_list.append(self.axs[0].axvspan(xmin, T[k] + self.window, facecolor=color, alpha=0.1))
                        xmin = T[k+1]
                else:
                    self.axvspan_list.append(self.axs[0].axvspan(xmin, T[k] + self.window, facecolor=color, alpha=0.1))
            self.axs[0].text(x=T[k] + self.window, y=0.5, s=str(cl.index(T)))

    def build_plot(self):
        # In case a cluster is already defined in ND view, show time-zones related to segmented clusters
        if len(self.clusters) > 0:
            self.update_colors(which="all")

        # Build color succession in graph
        cm = plt.get_cmap('tab20c')
        # Building a summary from raw data
        summary = self.original_unorm_dataframe.describe().T
        n_props = len(summary.T.columns)
        colors = [cm(1.*i/n_props) for i in range(n_props)]
        self.axs[0].set_prop_cycle(color=colors)

        # Raw plot of normalized data
        self.original_dataframe.plot(ax=self.axs[0], linewidth=2, legend=False)

        # Table statistics of raw data
        # recreate the scientific format for values
        scientific = [['{:.2e}'.format(v) for v in r] for r in summary.values]
        self.axs[1].table(cellText=scientific, colLabels=summary.columns, rowLabels=summary.T.columns,
                          rowColours=colors, cellLoc='center', loc='center')

        self.fig2.show()

    def selection_Nd_to_2d(self, cluster_Nd):
        # to be able to modify passed self.cluster directly
        cluster_2d = []
        cluster_2d_label = []
        for k in range(len(cluster_Nd)):
            cluster_in_slice = [self.index_2D[i] for i in cluster_Nd[k] if i in self.index_ND[self.layer]]
            if len(cluster_in_slice) > 0:
                cluster_2d += [cluster_in_slice]
                cluster_2d_label += [cluster_Nd.index(cluster_Nd[k])]
        return cluster_2d, cluster_2d_label


# TODO : think about a framework to compare two or more simulations like this
