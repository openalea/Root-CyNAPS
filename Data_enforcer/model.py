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

    def transportW(self, time):
        self.Total_Transpiration[1] = self.dataset["Total_Transpiration"][time]*(1e-3)*1e-4

    def transportN(self, time):
        axial_diffusion_xylem: float = 2.5e-4   # g.m-2.s-1
        axial_diffusion_phloem: float = 1e-4  # g.m-2.s-1
        shoot_xylem_Nm = 1e-5   # mol.g-1 DW
        shoot_xylem_AA = 1e-5   # mol.g-1 DW
        shoot_phloem_AA = 2e-3  # mol.g-1 DW
        xylem_cross_area_ratio: float = 0.84 * (0.36 ** 2)  # (adim) apoplasmic cross-section area ratio * stele radius ratio^2

        if self.Total_Transpiration[1] >= 0:
            Nm_water_conc = self.root_xylem_Nm[1] * self.collar_struct_mass[1] * xylem_cross_area_ratio / self.root_xylem_water[1]
            AA_water_conc = self.root_xylem_AA[1] * self.collar_struct_mass[1] * xylem_cross_area_ratio / self.root_xylem_water[1]

        else:
            Nm_water_conc = 1e-6
            AA_water_conc = 1e-6

        Nm_collar_advection = Nm_water_conc * self.Total_Transpiration[1]
        AA_collar_advection = AA_water_conc * self.Total_Transpiration[1]

        # note, gradients are not computed in the same way for xylem and phloem, we have an a priori on flow directions
        Nm_collar_xylem_diffusion = axial_diffusion_xylem * (self.root_xylem_Nm[1] - shoot_xylem_Nm) * np.pi * self.root_radius[1]**2
        AA_collar_xylem_diffusion = axial_diffusion_xylem * (self.root_xylem_AA[1] - shoot_xylem_AA) * np.pi * self.root_radius[1]**2

        AA_collar_phloem_diffusion = axial_diffusion_phloem * (shoot_phloem_AA - self.root_phloem_AA[1]) * np.pi * self.root_radius[1] ** 2

        self.Export_Nitrates[1] = Nm_collar_advection + Nm_collar_xylem_diffusion
        self.Export_Amino_Acids[1] = AA_collar_advection + AA_collar_xylem_diffusion

        self.Unloading_Amino_Acids[1] = AA_collar_phloem_diffusion

        self.Export_cytokinins[1] = self.dataset["Export_cytokinins"][time]

    def exchanges_and_balance(self, time):
        # Water flow first for advection computation
        self.transportW(time)
        self.transportN(time)
