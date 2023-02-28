"""
rhizodep.soil
_________________
This is the boundary soil module for rhizodep.

Documentation and features
__________________________


Main functions
______________
Dataclasses are used for MTG properties' initialization and parametrization.
Classes' names represent accounted hypothesis in the progressive development of the model.

Methods :
- ()
"""
# Imports
from dataclasses import dataclass
import numpy as np


# Mean soil concentrations

@dataclass
class MeanConcentrations:
    soil_Nm: float = 1e-3
    soil_AA: float = 1e-3

# External conditions parameters


@dataclass
class SoilPatch:
    soil_Nm_max: float = 0.01
    patch_dilution: float = 0.0001/3600
    z_soil_Nm_max: float = 0
    lixiviation_speed: float = 0.001/3600
    soil_Nm_variance: float = 0.0001


class SoilNitrogen:
    def __init__(self, g, soil_Nm, soil_AA):

        # New properties' creation in MTG
        keywords = dict(
            soil_Nm=soil_Nm,
            soil_AA=soil_AA)

        props = g.properties()
        for name in keywords:
            props.setdefault(name, {})

        # vertices storage for future calls in for loops
        self.vertices = g.vertices(scale=g.max_scale())
        for vid in self.vertices:
            for name, value in keywords.items():
                # Effectively creates the new property
                props[name][vid] = value
        
        # Accessing properties once, pointing to g for further modifications
        states = """
                    soil_Nm
                    soil_AA
                    z1
                    """.split()
        
        for name in states:
            setattr(self, name, props[name])

    def update_patches(self, patch_age, soil_Nm_max, patch_dilution, z_soil_Nm_max, lixiviation_speed, soil_Nm_variance):
        # for all root segments in MTG...

        # Patch intensity decreasing with time
        max_concentration =  soil_Nm_max - patch_dilution * patch_age

        # Patch lixiviation with time
        depth_of_max = z_soil_Nm_max - lixiviation_speed * patch_age

        for vid in self.vertices:
            self.soil_Nm[vid] = (
                        (max_concentration * np.exp(-((self.z1[vid] - depth_of_max) ** 2) / soil_Nm_variance))
                )


