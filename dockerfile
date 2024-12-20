# Specifying the base image
FROM condaforge/mambaforge:23.3.1-1

RUN mamba create -y -n root_bridges -c conda-forge -c openalea3 --strict-channel-priority --file requirements.txt

# May be optional
RUN mamba init bash

# Equivalent command to mamba activate root_cynaps
SHELL ["mamba", "run", "-n", "root_cynaps", "/bin/bash", "-c"]

RUN apt-get update && apt-get -y install libgl1

RUN mkdir pp

ADD . ./pp/root_cynaps

WORKDIR ./pp/root_cynaps

RUN python setup.py develop

VOLUME ./pp/root_cynaps/simulations/running_scenarios/outputs

# Mandatory for good packages indexing
USER root

CMD python simulations/running_scenarios/main_run_scenarios.py
