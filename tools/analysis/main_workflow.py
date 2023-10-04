'''IMPORTS'''
import os, shutil, importlib
import numpy as np
from keras.models import Model, load_model
import umap
import hdbscan

import tools.analysis.time_series_projection
from tools.analysis.time_series_projection import Preprocessing, DCAE

'''SCRIPT'''
input_type = "mtg"

# DCAE parameters
import_model, train_model = False, True
dev = True
window = 24
EPOCHS = 25
BS = 100
test_prop = 0.2
# UMAP Parameters
umap_seed = 42
umap_dim = 10
n_neighbors = 15
min_dist = 0.15
# HDBSCAN Parameters
min_cluster_size = 5000
min_samples = 10

# Coordinates in topology
mtg_coordinates = dict(
    distance_from_tip=dict(unit="m", value_example=0.026998706,
                           description="Distance between the root segment and the considered root axis tip")
)

# Properties of interest (Remove constant variables or training will fail)
# TODO actualize
flow_extracts = dict(
    import_Nm=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    export_Nm=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    export_AA=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    # diffusion_Nm_soil=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_Nm_xylem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    # diffusion_Nm_soil_xylem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    # diffusion_AA_soil=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_AA_phloem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    # diffusion_AA_soil_xylem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    AA_synthesis=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    struct_synthesis=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    storage_synthesis=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    AA_catabolism=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    # storage_catabolism=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    # Water model
    radial_import_water=dict(unit="mol H2O.s-1", value_example=float(0), description="not provided"),
    axial_export_water_up=dict(unit="mol H2O.s-1", value_example=float(0), description="not provided"),
    axial_import_water_down=dict(unit="mol H2P.s-1", value_example=float(0), description="not provided"),
)


def run_analysis(path, input_type=input_type, import_model=import_model, train_model=train_model, dev=dev, window=window,
                 EPOCHS=EPOCHS, BS=BS, test_prop=test_prop, umap_seed=umap_seed, umap_dim=umap_dim, n_neighbors=n_neighbors, min_dist=min_dist,
                 min_cluster_size=min_cluster_size, min_samples=min_samples,mtg_coordinates=mtg_coordinates,
                 flow_extracts=flow_extracts):
    # Import and preprocess data
    print("[INFO] preprocessing data...")
    preprocess = Preprocessing(df_path=path, type=input_type, coordinates=mtg_coordinates, variables=flow_extracts,
                               window=window)
    stacked_dataset = [np.array(k) for k in preprocess.stacked_dataset]

    if import_model:
        print("[INFO] loading autoencoder...")
        autoencoder = load_model("tools/analysis/saved_model/autoencoder")
    else:
        # Build the convolutional autoencoder
        print("[INFO] building autoencoder...")
        (encoder, decoder, autoencoder) = DCAE.build(height=1, width=window, depth=len(flow_extracts),
                                                     filters=((64, 10, 2), (32, 5, 2), (12, 5, 3)), latentDim=window)

    if train_model:
        print("[INFO] training autoencoder...")
        autoencoder = DCAE.train(stacked_dataset=stacked_dataset, autoencoder=autoencoder, test_prop=test_prop,
                                 epochs=EPOCHS, batch_size=BS)

        if input("Save autoencoder? WARNING Overwrite (y/n) : ") == 'y':
            folder = os.path.dirname(__file__)
            shutil.rmtree(folder + '/saved_model/autoencoder')
            os.mkdir(folder + '/saved_model/autoencoder')
            autoencoder.save(folder + '/saved_model/autoencoder')

    # use the convolutional autoencoder to predict latent layer from trained encoder only
    # in autoencoder, -2 to retrieve encoder, -1 for decoder
    trained_encoder = Model(autoencoder.input, autoencoder.layers[-2].output)
    # print(trained_encoder.summary())

    # project on latent representation for all dataset, here time windows are still ordered in the obtained array
    print("[INFO] encoding windows...")
    latent_windows = []
    for k in stacked_dataset:
        latent_windows += [trained_encoder.predict(k)]
    # print(latent_windows) # number of extracter windows x window size

    # Latent space projection on lower dimension
    print("[INFO] UMAP reducer processing...")
    umap_reducer_ND = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=umap_dim, random_state=umap_seed)

    print("     Whole latent windows...")
    windows_ND_embedding = umap_reducer_ND.fit_transform(np.concatenate(latent_windows))

    print("[INFO] HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40, metric='euclidean', min_cluster_size=min_cluster_size,
                                min_samples=min_samples, p=None)

    clusterer.fit(windows_ND_embedding)

    groups = list(np.unique(clusterer.labels_))
    hdbscan_clusters = [[] for k in groups]
    for c in range(len(clusterer.labels_)):
        hdbscan_clusters[groups.index(clusterer.labels_[c])] += [c]


    # A position translator is needed to pass selections between 3d+ and 2d representations
    index_2d = []
    index_Nd = []
    m = 0
    for k in latent_windows:
        index_2d += [i for i in range(len(k))]
        index_Nd += [[m + i for i in range(len(k))]]
        m += len(k)

    # Plotting projection

    stacked_dataframe = preprocess.stacked_dataframe
    stacked_unorm_dataframe = preprocess.stacked_unorm_dataframe
    if umap_dim != 3:
        plot = False
    else:
        plot = True

    # Using this loop to be able to implement visualization without re-running UMAP each time
    while dev:
        # If re-running, reload the local plotting library
        importlib.reload(tools.analysis.time_series_projection)
        from tools.analysis.time_series_projection import MainMenu

        main_menu = MainMenu(windows_ND_projection=windows_ND_embedding, latent_windows=latent_windows,
                             sliced_windows=stacked_dataset, original_unorm_dataframe=stacked_unorm_dataframe,
                             original_dataframe=stacked_dataframe, coordinates=preprocess.coord,
                             clusters=hdbscan_clusters, index_2D=index_2d, index_ND=index_Nd, window=window, plot=plot)
        main_menu.build_app()
        again = input("reimport and replot? (y/n)")
        if again != "y":
            dev = False

    # If this is user call of the analysis
    if not dev:
        main_menu = MainMenu(windows_ND_projection=windows_ND_embedding, latent_windows=latent_windows,
                             sliced_windows=stacked_dataset, original_unorm_dataframe=stacked_unorm_dataframe,
                             original_dataframe=stacked_dataframe, coordinates=preprocess.coord,
                             clusters=hdbscan_clusters, index_2D=index_2d, index_ND=index_Nd, window=window, plot=plot)
        main_menu.build_app()
