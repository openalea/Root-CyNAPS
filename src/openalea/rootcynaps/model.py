import src.openalea.rootcynaps.model as model

from rootcynaps.root_nitrogen import RootNitrogenModel
from rootcynaps.root_water import RootWaterModel
from root_bridges.soil_model import SoilModel
from rootcynaps.root_anatomy import RootAnatomy
from rhizodep.root_growth import RootGrowthModel

from openalea.metafspm.composite_wrapper import CompositeModel
from openalea.metafspm.component_factory import Choregrapher

from analyze.analyze import add_root_order_when_branching_is_wrong


class Model(CompositeModel):
    """
    Root-CyNAPS model

    Use guideline :
    1. store in a variable Model(g, time_step) to initialize the model, g being an openalea.MTG() object and time_step an time interval in seconds.

    2. print Model.documentation for more information about editable model parameters (optional).

    3. Use Model.scenario(**dict) to pass a set of scenario-specific parameters to the model (optional).

    4. Use Model.run() in a for loop to perform the computations of a time step on the passed MTG File
    """

    def __init__(self, name: str = "Plant", time_step: int = 3600, coordinates: list=[0, 0, 0], **scenario: dict):
        """
        DESCRIPTION
        ----------
        __init__ method of the model. Initializes the thematic modules and link them.

        :param g: the openalea.MTG() instance that will be worked on. It must be representative of a root architecture.
        :param time_step: the resolution time_step of the model in seconds.
        """

        # DECLARE GLOBAL SIMULATION TIME STEP
        self.name = name
        Choregrapher().add_simulation_time_step(time_step)

        parameters = scenario["parameters"]["root_cynaps"]["roots"]
        self.time = parameters["plant_age"]
        self.input_tables = scenario["input_tables"]

        # INIT INDIVIDUAL MODULES
        # Here we use the growth model simply to initialize the structural mass and distance from tip regarding provided MTG's geometry.
        self.root_growth = RootGrowthModel(scenario["input_mtg"]["root_mtg_file"], time_step, **parameters)
        self.g = self.root_growth.g
        add_root_order_when_branching_is_wrong(self.g)
        self.root_anatomy = RootAnatomy(self.g, time_step, **parameters)
        self.root_water = RootWaterModel(self.g, time_step, **parameters)
        self.root_nitrogen = RootNitrogenModel(self.g, time_step, **parameters)
        self.soil = SoilModel(self.g, time_step, **parameters)
        self.soil_voxels = self.soil.voxels

        # LINKING MODULES
        self.declare_data_and_couple_components(root=self.g, soil=self.soil_voxels,
                                                translator_path=model.__path__[0],
                                                components=(self.root_growth, self.root_anatomy, self.root_water, self.root_nitrogen, self.soil))

        # Some initialization must be performed after linking modules
        self.root_water.post_coupling_init()

        # Update topological surfaces and volumes based on initialized structural properties
        self.root_anatomy()
        self.soil()

    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.components, when=self.time)
        
        # Compute state variations for water and then carbon and nitrogen
        self.root_water()
        self.root_nitrogen()

        self.time += 1
