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
import xbatcher as xb
import numpy as np
# Visual packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import CenteredNorm
import tkinter as tk

# Tensor management packages
from tensorflow import keras, reshape
from keras.layers import MaxPool2D, Conv2D, Conv2DTranspose, ReLU, UpSampling2D, Activation, Flatten, Dense, Reshape, Input
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Projection
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import f_oneway
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from root_cynaps.tools_output import plot_xr

'''FUNCTIONS'''


class Preprocessing:
    def __init__(self, central_dataset, type='csv', variables={}, window=24, stride=12):
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
        depth = len(variables)
        self.stacked_da = np.concatenate([[reshape(batch, shape=(1, window, depth))] for batch in bgen])

    def normalization(self, dataset):
        """
        Standard normalization technique
        NOTE : per organ normalization was fucking up the relative magnitude of the different organ comparison
        Now, the standardization is operated from min and max for all t, vid and scenario parameter.
        Still it remains essential to be able to compare the magnitude of the differences between clusters
        """

        return (dataset - dataset.min()) / (dataset.max() - dataset.min())


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
    def __init__(self, windows_ND_projection, latent_windows, sliced_windows, original_unorm_dataset, original_dataset, coordinates, clusters, windows_time, window=60, plot=False, output_path=""):
        # Retrieving necessary dataset
        self.original_unorm_dataset = original_unorm_dataset
        self.original_dataset = original_dataset
        self.sliced_windows = sliced_windows
        self.latent_windows = latent_windows
        self.windows_ND_projection = windows_ND_projection

        self.properties = list(self.original_unorm_dataset.keys())
        self.coordinates = coordinates
        self.window = window
        self.windows_time = windows_time
        self.vid_numbers = np.unique([index[-1] for index in self.coordinates])
        self.sensitivity_coordinates = [index[:-1] for index in self.coordinates]

        # Tk init
        self.root = tk.Tk()
        self.root.title('ND UMAP projection of extracted windows')
        self.output_path = output_path

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

        svm_button = tk.Button(self.root, text='SVM comparison', command=self.svm_selection)
        svm_button.grid(row=2, column=3)

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

    def svm_selection(self):
        print("[INFO] Testing clusters significativity...")
        classes = []
        selected_groups = []
        if len(self.clusters) == 0:
            print("[Error] : No selection")
        elif len(self.clusters) == 1:
            self.clusters += [[k for k in range(len(self.windows_ND_projection)) if k not in self.clusters[0]]]
            self.update_colors()
        # Retrieving latent windows corresponding to selected groups
        for k in range(len(self.clusters)):
            classes += [k for j in range(len(self.clusters[k]))]
            selected_groups += [list(i) for i in self.latent_windows[self.clusters[k]]]

        if len(self.clusters) > 1:
            # splitting data for svm training and test
            x_train, x_test, y_train, y_test = train_test_split(selected_groups, classes, test_size=0.2)
            clf = SVC(kernel='linear', C=100)
            # Here we use all data because we just perform analysis of model performance at segmentation,
            # we don't want it to be predictive
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

    def compute_group_area_between_curves(self):
        # Check individual variables contributions to differences between clusters
        # for each labelled cluster
        nb_props = len(self.properties)

        # Back to original data
        # grouping separated variable windows to use indexes selected by clusters
        clusters_windows = [self.sliced_windows[k] for k in self.clusters]
        # For each of the clusters, for each variable, we compute the mean of values for t0, t1, ...
        # Index 0 is used for windows because that's how the DCAE input shape was designed (1 x window x nb_props)
        curve_sets_clusters = [[[[win[0][t][v] for win in clst] for t in range(self.window)] for v in range(nb_props)] for clst in clusters_windows]
        # (unit = sum(flux_unit*time_step) = quantity per time_step hours)

        # Matrix to present main responsible for divergence between clusters through the Area Under the Curve (AUC)
        abcs = [[{} for k in range(len(self.clusters))] for i in range(len(self.clusters))]
        mean_diff_bcs = [[{} for k in range(len(self.clusters))] for i in range(len(self.clusters))]
        # for each row
        for k in range(len(self.clusters)):
            # for each element after diagonal in row
            for l in range(k+1, len(self.clusters)):
                # For a given variable, get the differences of means at each time-step
                # Then sum this to compute Area Under the curve for each variable and add it in the cross comparison matrix
                # for each variable, label the sum to a name (dict key) for readability
                for v in range(nb_props):
                    step_wise_differences = [(np.mean(curve_sets_clusters[k][v][t]) - np.mean(curve_sets_clusters[l][v][t])) *
                                                        (1-f_oneway(curve_sets_clusters[k][v][t], curve_sets_clusters[l][v][t]).pvalue) for t in range(self.window)
                                                        if not np.isnan(f_oneway(curve_sets_clusters[k][v][t], curve_sets_clusters[l][v][t]).pvalue)]
                    abcs[k][l][self.properties[v]] = np.sum([abs(diff) for diff in step_wise_differences])
                    mean_diff_bcs[k][l][self.properties[v]] = np.mean(step_wise_differences) * self.window

        return abcs, mean_diff_bcs

    def cluster_info(self):
        print("[INFO] Comparing clusters...")
        if len(self.clusters) == 0:
            print("[Error] : no cluster selected")
            return

        abcs, mean_diff_between_clusters = self.compute_group_area_between_curves()

        fig3 = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, len(self.clusters), height_ratios=[1, 2], figure=fig3)

        ax30 = [fig3.add_subplot(gs[0, k]) for k in range(len(self.clusters))]
        ax31 = fig3.add_subplot(gs[1, :])

        fig3.text(0.01, 0.95, "Space-Time repartition", fontweight='bold')
        fig3.text(0.01, 0.50, "window ABC between clusters", fontweight='bold')

        heatmap = []
        heatmap_values = []
        pair_labels = []
        # for each cluster combination
        for k in range(len(self.clusters)):
            times = [self.windows_time[index] for index in self.clusters[k]]
            coords = [self.coordinates[index][-1] for index in self.clusters[k]]
            unique_vids = np.unique(coords)
            maxs_index = [k for k, v in sorted(dict(zip(unique_vids, [coords.count(k) for k in unique_vids])).items(),
                                               key=lambda item: item[1], reverse=True)]

            ax30[k].set_title("C" + str(k) + " : " + str(int(len(self.clusters[k])/1000)) + "k / " + str(maxs_index[1:4])[1:-1])
            ax30[k].hist2d(times, coords, bins=20, cmap="Purples")

            for i in range(k+1, len(self.clusters)):
                heatmap += [list(abcs[k][i].values())]
                heatmap_values += [list(mean_diff_between_clusters[k][i].values())]
            pair_labels += ["{}-{}".format(k, i) for i in range(k + 1, len(self.clusters))]

        hm = ax31.imshow(np.transpose(heatmap), cmap="Greens", aspect="auto", vmin=0)
        fig3.colorbar(hm, orientation='horizontal', location='top')

        ax31.set_xticks(np.arange(len(pair_labels)), labels=pair_labels)
        ax31.set_yticks(np.arange(len(self.properties)), labels=self.properties)

        # Loop over data dimensions and create text annotations.
        for i in range(len(pair_labels)):
            for j in range(len(self.properties)):
                ax31.text(i, j, round(heatmap_values[i][j], 2), ha="center", va="center", color="w",
                               fontsize=10, fontweight='bold')

        fig3.set_size_inches(19, 10)
        fig3.savefig(self.output_path + "/clustering.png", dpi=400)
        fig3.show()

    def flat_plot_instance(self):
        layer = self.lb.curselection()
        if len(layer) > 0:
            layer = self.vid_numbers[layer[0]]
            plot_xr(datasets=self.original_unorm_dataset, vertice=[layer], selection=self.properties)

    def cluster_sensitivity_test(self, alpha=0.05):
        # Starting with multivariate anova assuming normality
        # Dataframe formating...
        classes = []
        selected_groups = []
        # Tuple is necessary here because this call is "Frozen"
        sensi_names = tuple(dim for dim in self.original_unorm_dataset.dims.keys() if dim not in ("t", "vid"))
        for c in range(len(self.clusters)):
            classes += [str(c) for j in range(len(self.clusters[c]))]
            selected_groups += [self.sensitivity_coordinates[k] for k in self.clusters[c]]
        cluster_sensi_values = pd.DataFrame(data=selected_groups, columns=sensi_names)
        cluster_sensi_values['cluster'] = classes

        # MANOVA for sensitivity factors across the cluster factor
        sensi_sum = ""
        for name in sensi_names:
            sensi_sum += f"{name} + "
        sensi_sum = sensi_sum[:-3]

        fit = MANOVA.from_formula(f'{sensi_sum} ~ cluster', data=cluster_sensi_values)
        manova_df = pd.DataFrame((fit.mv_test().results['cluster']['stat']))
        manova_pv = float(manova_df.loc[["Wilks' lambda"]]["Pr > F"])

        # If there is a significant difference between clusters regarding sensitivity variables...
        if manova_pv < alpha:
            # Perform pairwise tukey post-hoc test to identify which clusters are different
            meandiff_line = []
            significativity = []
            for sensi in sensi_names:
                tuckey_test = pairwise_tukeyhsd(cluster_sensi_values[sensi], cluster_sensi_values['cluster'], alpha=alpha).summary().data
                column_names = tuckey_test[0]
                pairwise_label = [line[column_names.index('group1')] + '-' + line[column_names.index('group2')] for line in tuckey_test[1:]]
                meandiff_line += [[line[column_names.index('meandiff')] for line in tuckey_test[1:]]]
                significativity += [[str(line[column_names.index('reject')]) for line in tuckey_test[1:]]]

            fig_tuckey, ax = plt.subplots()
            ax.set_xticks(np.arange(len(pairwise_label)), labels=pairwise_label)
            ax.set_yticks(np.arange(len(sensi_names)), labels=sensi_names)
            shifted_colormap = CenteredNorm()
            hm = ax.imshow(meandiff_line, cmap="PiYG", aspect="auto", norm=shifted_colormap)
            fig_tuckey.colorbar(hm, orientation='horizontal', location='top')
            # Loop over data dimensions and create text annotations.
            for i in range(len(sensi_names)):
                for j in range(len(pairwise_label)):
                    ax.text(j, i, significativity[i][j], ha="center", va="center", color="b",
                              fontsize=10, fontweight='bold')
            fig_tuckey.set_size_inches(19, 10)
            fig_tuckey.savefig(self.output_path + "/pairwise_tucker.png", dpi=400)
            fig_tuckey.show()

            significant_sensitivity = [False]
        else:
            significant_sensitivity = [False]
        return significant_sensitivity

    def plantGL_map(self):
        # TODO
        return

    def build_app(self, plot=False):
        # time is an axis as others, rather, default color corresponding coordinates on structure
        # For a given cluster, it wil enable user to select 2D plots of interest for targeted layers, WITH corresponding clusters highlighted (and refreshed)

        self.cluster_info()
        self.cluster_sensitivity_test()
        self.plantGL_map()
        self.root.mainloop()
