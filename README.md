# Root-CyNAPS : Root Cycling Nitrogen Across Plant Scales
### A 3D FSPM root model to simulate nitrogen acquisition and rhizodeposition

[![CI status](https://github.com/openalea/root-cynaps/actions/workflows/conda-package-build.yml/badge.svg)](https://github.com/openalea/root-cynaps/actions/workflows/conda-package-build.yml)
[![Documentation Status](https://readthedocs.org/projects/root-cynaps/badge/?version=latest)](https://root-cynaps.readthedocs.io/en/latest/?badge=latest)
[![image](https://anaconda.org/openalea3/openalea.rootcycnaps/badges/version.svg)](https://anaconda.org/openalea3/openalea.rootcynaps)

## Software

### Authors

> -   Tristan Gérault
> -   Christophe Pradal
> -   Romain Barillot
> -   Céline Richard-Molard
> -   Marion Gauthier
> -   Alexandra Jullien
> -   Frédéric Rees

### Institutes

INRAE / AgroParisTech / CIRAD / INRIA

### Status

Python package

### License

CecILL-C

**URL** : <https://root-cynaps.rtfd.io>

## About

### Description

This module aims at simulating explicit processes of root nitrogen uptake and rhizodeposition by aggregating a segment-scale nutrient balance model over the root system architecture.

### Content

The OpenAlea.HydroRoot package contains a root architecture simulation
model coupled with a water and solute transport solver. It contains a
pure hydraulic solver that is solved using resistance network analogy.
It also contains a water and solute transport solver that is more
complex and see the root as a continuous medium.

### Example

Heat map representation of the incoming local radial flows on an
arabidopsis root.

![Alt Text](example/data/fig-6E.png)

### Installation

## Installation option 1 : stable installation with docker (about 6Gb)

1. First, download the docker engine : https://docs.docker.com/engine/install/

2. Then create a project directory (example : PythonProjects), and clone this repository from command line :
```
cd .../PythonProjects
git clone https://forgemia.inra.fr/tristan.gerault/root_cynaps.git
```

3. Finally, build the docker image with the following command
```
cd root_cynaps
docker image build -t root_cynaps .
docker run root_cynaps
```

(For developpers, build this image directly into your IDE)'


## Installation option 2 : more classic but unstable installation with conda + mamba (about 4Gb)



Intall the lastest version of conda : https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html


#### Conda Installation

    mamba create -n root-cynaps -c conda-forge -c openalea3 openalea.rootcynaps

#### Developper Installation

First, clone or fork the repository to retreive source code :
```
git clone https://github.com/openalea/root-cynaps.git
```

Then create a conda environment in which to install Root-CyNAPS requirements into
```
mamba create -n root-cynaps
```

Finally to install model's dependencies, 1) open a terminal 2) 


#### Requirements

> -   openalea.mtg
> -   openalea.plantgl
> -   RSML
> -   pandas \> 0.17
> -   numpy
> -   scipy

#### Usage

See notebook in example directory.
cd to the simulations/running_example folder, then run :
```
python simulation.py
```

## Documentation

<https://hydroroot.rtfd.io>

## Citations

RootCyNAPS has been submitted for publication. For now, please cite:

    Tristan Gérault, Christophe Pradal, Romain Barillot, Céline Richard-Molard, Marion Gauthier, Alexandra Jullien, Frédéric Rees, 
    "Do specific root zones shape plant net N uptake? Modeling insights from Root-CyNAPS", submitted