:: Model packages
conda install -y -c conda-forge mamba

mamba create -n root_cynaps python=3.9
mamba activate root_cynaps
mamba install -y -c openalea3 -c conda-forge openalea.plantgl openalea.mtg

:: Data analysis and result visualisation packages
mamba install -y -c conda-forge xarray==2023.3.0 dask==2023.3.2 bottleneck==1.3.7
python -m pip install netcdf4==1.6.3
python -m pip install --force-reinstall charset-normalizer==3.1.0
python -m pip install pandas==1.5.3
python -m pip install matplotlib==3.7.0
python -m pip install scikit-learn==1.2.2
python -m pip install numpy==1.22.4
mamba install -y -c conda-forge umap-learn==0.5.3
mamba install -y -c conda-forge hdbscan==0.8.29
python -m pip install tensorflow==2.12.0
python -m pip install pyncclient


