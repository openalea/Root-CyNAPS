# Faking retrival of flows computed by another shoot model that should be built as a dict
from dataclasses import dataclass, asdict


nitrogen_flows = {
    "AA_root_shoot_phloem": "Unloading_Amino_Acids",
    "cytokinins_root_shoot_xylem": "Export_cytokinins"
}

water_flows = {
    "water_root_shoot_xylem": "Total_Transpiration"
}

nitrogen_state = {
    "root_xylem_Nm": "xylem_Nm",
    "root_xylem_AA": "xylem_AA",
    "collar_struct_mass": "struct_mass",
    "root_phloem_AA": "phloem_AA",
    "root_radius": "radius",
    "segment_length": "radius"
}

water_state = {
    "root_xylem_water": "xylem_water",
    "root_xylem_pressure": "xylem_total_pressure"
}


def link_mtg(receiver, applier, category, translator={}, same_names=True):
    """
    Description : linker function that will enable properties sharing through MTG.

    Parameters :
    :param receiver: (class) model class whose inputs should be provided with the applier class.
    :param applier: (class) model class whose properties are used to provide inputs to the receiver class.
    :param category: (sting) word to specify which inputs are to be considered in the receiver model class.
    :param translator: (dict) translation dict used when receiver and applier properties do not have the same names.
    :param same_names: (bool) boolean value to be used if a model was developped by another team with different names.

    Note :  The whole property is transfered, so if only the collar value of a spatial property is needed,
    it will be accessed through the first vertice with the [1] indice. Not spatialized properties like xylem pressure or
    single point properties like collar flows are only stored in the indice [1] vertice.
    """
    if same_names:
        for link in getattr(receiver, "inputs")[category]:
            setattr(receiver, link, getattr(applier, link))
    else:
        for link in getattr(receiver, "inputs")[category]:
            setattr(receiver, link, getattr(applier, translator[link]))

