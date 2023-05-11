# Faking retrival of flows computed by another shoot model that should be built as a dict
from dataclasses import dataclass, asdict


nitrogen_flows = {
    "Nm_root_shoot_xylem":"Nm_root_shoot_xylem",
    "AA_root_shoot_xylem":"AA_root_shoot_xylem",
    "AA_root_shoot_phloem":"AA_root_shoot_phloem",
    "cytokinins_root_shoot_xylem":"cytokinins_root_shoot_xylem"
}

water_flows = {
    "water_root_shoot_xylem":"water_root_shoot_xylem"
}

nitrogen_state = {
    "root_xylem_Nm":"xylem_Nm",
    "root_xylem_AA":"xylem_AA",
    "collar_struct_mass":"struct_mass",
    "root_phloem_AA":"phloem_AA",
    "root_radius":"radius",
    "segment_length":"radius"
}

water_state = {
    "root_xylem_water":"xylem_water",
    "root_xylem_pressure":"xylem_total_pressure"
}

def link_mtg(reciever, applier, category, translator={}, same_names=True):
    if same_names:
        for link in getattr(reciever, "inputs")[category]:
            setattr(reciever, link, getattr(applier, link))
    else:
        for link in getattr(reciever, "inputs")[category]:
            setattr(reciever, link, getattr(applier, translator[link]))

def link_collar(reciever, applier, category, translator={}, same_names=True):
    if same_names:
        for link in getattr(reciever, "inputs")[category]:
            source = getattr(applier, link)
            if isinstance(source, dict):
                setattr(reciever, link, source.get(1))
            else:
                setattr(reciever, link, source)
    else:
        for link in getattr(reciever, "inputs")[category]:
            source = getattr(applier, translator[link])
            if isinstance(source, dict):
                setattr(reciever, link, source.get(1))
            else:
                setattr(reciever, link, source)
