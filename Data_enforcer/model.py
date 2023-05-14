import numpy as np
from dataclasses import dataclass, asdict
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
    def __init__(self, Nm_root_shoot_xylem, AA_root_shoot_xylem, AA_root_shoot_phloem, cytokinins_root_shoot_xylem,
                 water_root_shoot_xylem):
        self.keywords = dict(
            Nm_root_shoot_xylem=[Nm_root_shoot_xylem],
            AA_root_shoot_xylem=[AA_root_shoot_xylem],
            AA_root_shoot_phloem=[AA_root_shoot_phloem],
            cytokinins_root_shoot_xylem=[cytokinins_root_shoot_xylem],
            water_root_shoot_xylem=[water_root_shoot_xylem]
        )

        for name in self.keywords:
            setattr(self, name, self.keywords[name])

        self.inputs = {
            "root_nitrogen":[
                "root_xylem_Nm",
                "root_xylem_AA",
                "collar_struct_mass",
                "root_phloem_AA",
                "root_radius",
                "segment_length"],
            "root_water":[
                "root_xylem_water",
                "root_xylem_pressure"
            ]
        }

    def transportN(self, time_step):
        axial_diffusion_xylem: float = 2.5e-4   # g.m-2.s-1
        axial_diffusion_phloem: float = 1e-4  # g.m-2.s-1
        shoot_xylem_Nm = 1e-5   # mol.g-1 DW
        shoot_xylem_AA = 1e-5   # mol.g-1 DW
        shoot_phloem_AA = 2e-3  # mol.g-1 DW
        xylem_cross_area_ratio: float = 0.84 * (0.36 ** 2)  # (adim) apoplasmic cross-section area ratio * stele radius ratio^2

        if self.water_root_shoot_xylem[0] >= 0:
            Nm_water_conc = self.root_xylem_Nm[1] * self.collar_struct_mass[1] * xylem_cross_area_ratio / self.root_xylem_water[1]
            AA_water_conc = self.root_xylem_AA[1] * self.collar_struct_mass[1] * xylem_cross_area_ratio / self.root_xylem_water[1]

        else:
            Nm_water_conc = 1e-6
            AA_water_conc = 1e-6

        Nm_collar_advection = Nm_water_conc * self.water_root_shoot_xylem[0]
        AA_collar_advection = AA_water_conc * self.water_root_shoot_xylem[0]

        # note, gradients are not computed in the same way for xylem and phloem, we have an a priori on flow directions
        Nm_collar_xylem_diffusion = axial_diffusion_xylem * (self.root_xylem_Nm[1] - shoot_xylem_Nm) * np.pi * self.root_radius[1]**2
        AA_collar_xylem_diffusion = axial_diffusion_xylem * (self.root_xylem_AA[1] - shoot_xylem_AA) * np.pi * self.root_radius[1]**2

        AA_collar_phloem_diffusion = axial_diffusion_phloem * (shoot_phloem_AA - self.root_phloem_AA[1]) * np.pi * self.root_radius[1] ** 2

        self.Nm_root_shoot_xylem[0] = Nm_collar_advection + Nm_collar_xylem_diffusion
        self.AA_root_shoot_xylem[0] = AA_collar_advection + AA_collar_xylem_diffusion

        self.AA_root_shoot_phloem[0] = AA_collar_phloem_diffusion

        self.cytokinins_root_shoot_xylem[0] = 0

    def transportW(self, time_step):
        shoot_xylem_pressure = -2e6  # (Pa)
        sap_viscosity = 1.3e6
        # only hydrostatic for tests
        self.water_root_shoot_xylem[0] = ((np.pi * (self.root_radius[1]**4))/(8*sap_viscosity)) * (self.root_xylem_pressure[0] - shoot_xylem_pressure) / self.segment_length[1]

    def exchanges_and_balance(self, time_step):
        # Water flow first for advection computation
        self.transportW(time_step)
        self.transportN(time_step)