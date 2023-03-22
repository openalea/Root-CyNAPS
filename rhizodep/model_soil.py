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
    soil_water_pressure: float = 10325 # (Pa) mean soil water pressure
    soil_temperature: float = 283.15 # (K) mean soil temperature
    soil_Nm: float = 1e-3
    soil_AA: float = 1e-3

# External conditions parameters


@dataclass
class SoilPatch:
    patch: bool = False
    soil_Nm_max: float = 0.01
    patch_dilution: float = 0
    z_soil_Nm_max: float = 0
    lixiviation_speed: float = 0
    soil_Nm_variance: float = 1


class HydroMinSoil:
    def __init__(self, g, soil_water_pressure, soil_temperature, soil_Nm, soil_AA):

        # New properties' creation in MTG
        keywords = dict(
            soil_water_pressure=soil_water_pressure, # soil water content could be added to relate to water pressure
            soil_temperature=soil_temperature,
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
                    z1
                    """.split()
        # soil water pressure, temperature and AA are not imported here because we keep it constant for now
        
        for name in states:
            setattr(self, name, props[name])

    def update_patches(self, patch_age, patch, soil_Nm_max, patch_dilution, z_soil_Nm_max, lixiviation_speed, soil_Nm_variance):
        # if chosen option is homegeneous conditions
        if not patch:
            # for all root segments in MTG...
            for vid in self.vertices:
                self.soil_Nm[vid] = soil_Nm_max

        else:
            # Patch intensity decreasing with time
            max_concentration =  soil_Nm_max - patch_dilution * patch_age

            # Patch lixiviation with time
            depth_of_max = z_soil_Nm_max - lixiviation_speed * patch_age
            # for all root segments in MTG...
            for vid in self.vertices:
                self.soil_Nm[vid] = (max_concentration * np.exp(-((self.z1[vid] - depth_of_max) ** 2) / soil_Nm_variance))



