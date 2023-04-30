# Faking retrival of flows computed by another shoot model that should be built as a dict
from dataclasses import dataclass, asdict


@dataclass
class NitrogenComFlows:
    Nm_root_shoot_xylem: str = "Nm_root_shoot_xylem"
    AA_root_shoot_xylem: str = "AA_root_shoot_xylem"
    AA_root_shoot_phloem: str = "AA_root_shoot_phloem"
    cytokinins_root_shoot_xylem: str = "cytokinins_root_shoot_xylem"


@dataclass
class WaterComFlows:
    water_root_shoot_xylem: str = "water_root_shoot_xylem"


@dataclass
class ComState:
    xylem_Nm: str = "root_xylem_Nm"
    xylem_AA: str = "root_xylem_AA"
    struct_mass: str = "collar_struct_mass"
    xylem_water: str = "root_xylem_water"
    xylem_total_pressure: str = "root_xylem_pressure"
    phloem_AA: str = "root_phloem_AA"
    radius: str = "root_radius"
    length: str = "segment_length"


def apply_root_collar_flows(collar_flows, root_class, key):
    communication = {"nitrogen": asdict(NitrogenComFlows()),
                     "water": asdict(WaterComFlows())}
    com_table = communication[key]
    for name in com_table:
        setattr(root_class, name, collar_flows[com_table[name]])


def get_root_collar_state(root_class):
    class CollarState(object): pass
    collar_state = CollarState()
    communication = asdict(ComState())
    # get attribute names from the root class
    var = root_class.__dict__.keys()
    for name in communication:
        if name in var:
            target = getattr(root_class, name)
            # if it is a regular MTG property
            if isinstance(target, dict):
                setattr(collar_state, communication[name], target.get(1))
            # if it is an attribute summed over MTG
            else:
                setattr(collar_state, communication[name], target)
    return collar_state.__dict__
