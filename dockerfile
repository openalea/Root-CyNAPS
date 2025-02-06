# Specifying the base image
FROM condaforge/mambaforge:23.3.1-1

RUN mamba create -y -n root_bridges -c conda-forge -c openalea3 --strict-channel-priority --file requirements.txt

# May be optional
RUN mamba init bash

# Equivalent command to mamba activate root_cynaps
SHELL ["mamba", "run", "-n", "root_cynaps", "/bin/bash", "-c"]

RUN apt-get update && apt-get -y install libgl1

RUN mkdir pp

ADD . ./package/root_cynaps

WORKDIR ./package/root_cynaps

RUN python setup.py develop

VOLUME ./package/root_cynaps/simulations/outputs

# Mandatory for good packages indexing
USER root

CMD python -m ./package/root_cynaps/simulations/simutation.py
