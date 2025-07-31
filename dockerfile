# Specifying the base image
FROM condaforge/mambaforge

# May be optional
RUN mamba init bash

# Install directly in base to avoid a too large image
# Equivalent command to mamba activate base, just to be sure
SHELL ["mamba", "run", "-n", "base", "/bin/bash", "-c"]

RUN mkdir /package

WORKDIR /package

RUN git clone https://github.com/openalea/Root-CyNAPS.git
RUN git clone https://github.com/openalea/metafspm.git
RUN git clone https://github.com/openalea/fspm-utility.git

# Install local packages and their dependencies
WORKDIR /package/Root-CyNAPS
RUN mamba env create -n rootcynaps -f conda/environment.yaml
RUN pip install .
WORKDIR /package/metafspm
RUN pip install .
RUN mamba env update -n base -f conda/environment.yaml
WORKDIR /package/fspm-utility
RUN pip install .
RUN mamba env update -n base -f conda/environment.yaml

# Notebook to interact with the container
RUN mamba install -y -n rootcynaps -c conda-forge jupyterlab 

# To remove installation leftovers that might take too much space in the image
RUN conda clean -a -y

# graphical library necessary for simulations
RUN apt update && apt -y install libgl1-mesa-glx xvfb 

# Expose notebook port
EXPOSE 8888

# Set working directory to volume mount point
WORKDIR /data/notebooks

# Mandatory for good packages indexing
USER root

CMD ["mamba", "run", "-n", "rootcynaps", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--notebook-dir=/package/Root-CyNAPS/doc/notebooks", "/package/Root-CyNAPS/doc/notebooks/example_notebook_24h_static.ipynb"]
