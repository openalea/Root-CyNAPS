# Root-CyNAPS : Root Cycling Nitrogen Across Plant Scales
#### An architectured FSPM root model to simulate nitrogen acquisition and rhizodeposition

## Overview

This module aims at simulating explicit processes of root nitrogen uptake and rhizodeposition by agregating a segment scale nutrient balance to root an architectured root system scale

## Installation

First, clone the repository :
```
git clone https://forgemia.inra.fr/tristan.gerault/root_cynaps.git
```

Intall the lastest version of conda : https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

Then, create a new repository in your command prompt
```
conda create -n root_cynaps python=3.9
```

Finally, install necessary packages with the following requirements files. cd to the root_cynaps directory, then :

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

## Running example code

cd to the example folder, then run :
```
python main.py
```