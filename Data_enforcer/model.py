import numpy as np
from dataclasses import dataclass
import pandas as pd


@dataclass
class InitShootNitrogen:
    Nm_root_shoot_xylem: float = 0
    AA_root_shoot_xylem: float = 0
    AA_root_shoot_phloem: float = 0
    cytokinins_root_shoot_xylem: float = 0

@dataclass
class InitShootWater:
    water_root_shoot_xylem: float = 0


class ShootModel:
    def __init__(self, g, Nm_root_shoot_xylem, AA_root_shoot_xylem, AA_root_shoot_phloem, cytokinins_root_shoot_xylem,
                 water_root_shoot_xylem):

        self.g = g

        self.dataset = pd.read_csv("C:\\Users\\tigerault\\PythonProjects\\RHydroMin\\Data_enforcer\\inputs\\cnwheat_outputs.csv", sep=";")
        self.dataset = self.dataset.set_index("t")

        props = self.g.properties()
        for name in self.dataset.columns:
            props.setdefault(name, {})
            props[name][1] = self.dataset[name][0]
            setattr(self, name, props[name])

        self.inputs = {
            "root_nitrogen": [
                "root_xylem_Nm",
                "root_xylem_AA",
                "collar_struct_mass",
                "root_phloem_AA",
                "root_radius",
                "segment_length"],
            "root_water": [
                "root_xylem_water",
                "root_xylem_pressure"
            ]
        }

    def transport(self, time):
        for name in self.dataset.columns:
            print(name)
            getattr(self, name)[1] = self.dataset[name][time]*1e-6
            print(getattr(self, name))

    def exchanges_and_balance(self, time):
        # Water flow first for advection computation
        self.transport(time)
