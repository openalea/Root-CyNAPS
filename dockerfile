# Specifying the base image
FROM condaforge/mambaforge:23.3.1-1

RUN mamba create -y -n rootcynaps -c conda-forge -c openalea3 openalea.rootcynaps

# May be optional
RUN mamba init bash

# Equivalent command to mamba activate root_cynaps
SHELL ["mamba", "run", "-n", "root_cynaps", "/bin/bash", "-c"]

# graphical library necessary for simulations
RUN apt update && apt -y install libgl1

RUN mkdir rootcynaps_outputs

WORKDIR ./package/root_cynaps

RUN python setup.py develop

VOLUME ./package/root_cynaps/simulations/outputs

# Mandatory for good packages indexing
USER root

CMD python -m ./package/root_cynaps/simulations/simutation.py
