# Root-CyNAPS : Root Cycling Nitrogen Across Plant Scales
#### An architectured FSPM root model to simulate nitrogen acquisition and rhizodeposition

## Overview

This module aims at simulating explicit processes of root nitrogen uptake and rhizodeposition by aggregating a segment-scale nutrient balance model over the root system architecture.

TODO

## Installation option 1 : stable installation with docker

1. First, download the docker engine : https://docs.docker.com/engine/install/

2. Then create a project directory (example : PythonProjects), and clone this repository from command line :
```
cd .../PythonProjects
git clone https://forgemia.inra.fr/tristan.gerault/root_cynaps.git
```

3. Next, copy the file located in the root_cynaps folder, named "dockerfile" to the PythonProjects folder

```
# On windows :
copy .../PythonProjects/root_cynaps/dockerfile .../PythonProjects 
# On linux :
cp .../PythonProjects/root_cynaps/dockerfile .../PythonProjects 
```

4. Finally, build the docker image with the following command
```
docker image build -t root_cynaps .
docker run root_cynaps
```


## Installation option 2 : more classic but unstable installation with conda + mamba

First, clone the repository :
```
git clone https://forgemia.inra.fr/tristan.gerault/root_cynaps.git
```

Intall the lastest version of conda : https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

Then, create a new repository in your command prompt
```
conda create -n root_cynaps python=3.9
```

Then, install necessary packages with the following requirements files. cd to the root_cynaps directory, then :

- On Windows :
```
requirements.bat
```
- On Linux :
```
bash requirements.sh
```
- On MacOs :
No available yet

#### Running example code

cd to the simulations/running_example folder, then run :
```
python main.py
```