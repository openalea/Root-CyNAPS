import root_cynaps

from root_cynaps.root_nitrogen import RootNitrogenModel
from root_cynaps.root_water import RootWaterModel
from root_bridges.soil_model import SoilModel
from rhizodep.root_anatomy import RootAnatomy
from rhizodep.root_growth import RootGrowthModel

from metafspm.composite_wrapper import CompositeModel
from metafspm.component_factory import Choregrapher


class Model(CompositeModel):
    """
    Root-CyNAPS model

    Use guideline :
    1. store in a variable Model(g, time_step) to initialize the model, g being an openalea.MTG() object and time_step an time interval in seconds.

    2. print Model.documentation for more information about editable model parameters (optional).

    3. Use Model.scenario(**dict) to pass a set of scenario-specific parameters to the model (optional).

    4. Use Model.run() in a for loop to perform the computations of a time step on the passed MTG File
    """

    def __init__(self, time_step: int, **scenario: dict):
        """
        DESCRIPTION
        ----------
        __init__ method of the model. Initializes the thematic modules and link them.

        :param g: the openalea.MTG() instance that will be worked on. It must be representative of a root architecture.
        :param time_step: the resolution time_step of the model in seconds.
        """

        # DECLARE GLOBAL SIMULATION TIME STEP
        Choregrapher().add_simulation_time_step(time_step)
        self.time = scenario["parameters"]["root_cynaps"]["plant_age"]

        parameters = scenario["parameters"]["root_cynaps"]
        self.input_tables = scenario["input_tables"]

        # INIT INDIVIDUAL MODULES
        # Here we use the growth model simply to initialize the structural mass and distance from tip regarding provided MTG's geometry.
        self.root_growth = RootGrowthModel(scenario["input_mtg"]["root_mtg_file"], time_step, **parameters)
        self.g = self.root_growth.g
        self.root_anatomy = RootAnatomy(self.g, time_step, **parameters)
        self.root_water = RootWaterModel(self.g, time_step/10, **parameters)
        self.root_nitrogen = RootNitrogenModel(self.g, time_step, **parameters)
        self.soil = SoilModel(self.g, time_step, **parameters)
        self.soil_voxels = self.soil.voxels

        # ORDER MATTERS HERE !
        self.models = (self.soil, self.root_water, self.root_nitrogen, self.root_anatomy, self.root_growth)
        self.data_structures = {"root": self.g, "soil": self.soil_voxels}

        # LINKING MODULES
        self.link_around_mtg(translator_path=root_cynaps.__path__[0])

        # Some initialization must be performed after linking modules
        self.root_water.post_coupling_init()
        # Update topological surfaces and volumes based on initialized structural properties
        self.root_anatomy()
        self.soil()

    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.models, when=self.time)
        
        # Compute state variations for water and then carbon and nitrogen
        self.root_water()
        self.root_nitrogen()

        self.time += 1
