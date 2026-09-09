#!/bin/bash

### Les commentaires qui commencent par #$ sont interprétés par SGE comme des options en ligne ###
### ce script est un exemple, il est nécessaire de le modifier pour l'adapter à vos besoins ###

# Shell à utiliser pour l'exécution du job
#$ -S /bin/bash

# Nom du job
#$ -N rc_revision1

# Nom de la queue
#$ -q short.q
# -q long.q
# -q highmem.q

# Sélection d'un noeud particulier (commented)
# -l hostname=n12

# Sortie standard (already handled by python)
#$ -o /home/tigerault/work/Wheat-BRIDGES_framework/Root-CyNAPS/simulations/scene/outputs/output2.out

# Sortie d'erreur (already handled by python)
#$ -e /home/tigerault/work/Wheat-BRIDGES_framework/Root-CyNAPS/simulations/scene/outputs/errors2.err

# Mail
#$ -m ea
#$ -M tristan.gerault@inrae.fr

# Lance la commande depuis le répertoire où est lancé le script
#$ -cwd

# Utiliser n CPUs
#$ -pe thread 4

# Python

conda activate wheat-bridges
python -m analyze_simulation
conda deactivate