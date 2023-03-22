import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class InitShootNitrogen:
    Nm_root_shoot_xylem:float = 0
    AA_root_shoot_xylem:float = 0
    AA_root_shoot_phloem:float = 0
    cytokinins_root_shoot_xylem:float = 0

@dataclass
class InitShootWater:
    water_root_shoot_xylem:float = 0

@dataclass
class WTransport:
    axial_water_conductivity:float = 0.001

class ShootModel:
    def __init__(self, Nm_root_shoot_xylem, AA_root_shoot_xylem, AA_root_shoot_phloem, cytokinins_root_shoot_xylem,
                 water_root_shoot_xylem):
        self.keywords = dict(
            Nm_root_shoot_xylem=Nm_root_shoot_xylem,
            AA_root_shoot_xylem=AA_root_shoot_xylem,
            AA_root_shoot_phloem=AA_root_shoot_phloem,
            cytokinins_root_shoot_xylem=cytokinins_root_shoot_xylem,
            water_root_shoot_xylem=water_root_shoot_xylem
        )

        for name in self.keywords:
            setattr(self, name, self.keywords[name])

    def transportN(self):

        self.Nm_root_shoot_xylem = 0
        self.AA_root_shoot_xylem = 0
        self.AA_root_shoot_phloem = 0
        self.cytokinins_root_shoot_xylem = 0

        # Output flows
        class NFlows(object): pass
        N_flows = NFlows()
        N_flows.Nm_root_shoot_xylem = self.Nm_root_shoot_xylem
        N_flows.AA_root_shoot_xylem = self.AA_root_shoot_xylem
        N_flows.AA_root_shoot_phloem = self.AA_root_shoot_phloem
        N_flows.cytokinins_root_shoot_xylem = self.cytokinins_root_shoot_xylem
        return N_flows.__dict__

    def transportW(self, axial_water_conductivity, root_xylem_pressure, root_radius):
        shoot_xylem_pressure = 100000
        # only hydrostatic for tests
        self.water_root_shoot_xylem = axial_water_conductivity * (root_xylem_pressure - shoot_xylem_pressure) * np.pi * root_radius**2

        # Output flows
        class WFlows(object): pass
        W_flows = WFlows()
        W_flows.water_root_shoot_xylem = self.water_root_shoot_xylem
        return W_flows.__dict__

    def exchanges_and_balance(self, root_Nm, root_xylem_pressure, root_radius):
        N_flows = self.transportN()
        W_flows = self.transportW(root_xylem_pressure=root_xylem_pressure, root_radius=root_radius, **asdict(WTransport()))
        return N_flows, W_flows