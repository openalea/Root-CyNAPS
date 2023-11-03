from tools.analysis import main_workflow

def run(file, output_path):
    """
    :param path: specify path to xarray netcdf output files from mtg logging
    :return:
    """
    main_workflow.run_analysis(file, output_path, EPOCHS=10, min_cluster_size=5500)