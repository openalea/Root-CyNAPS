# Faking retrival of flows computed by another shoot model that should be built as a dict

def root_shoot_converter(model_path):
    collar_transports = dict(
        Nm_root_shoot_xylem=0,
        AA_root_shoot_xylem=0,
        AA_root_shoot_phloem=0,
        cytokinins_root_shoot_xylem=0)

    return collar_transports
