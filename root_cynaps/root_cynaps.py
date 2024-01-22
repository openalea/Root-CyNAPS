import yaml
import sysconfig

import root_cynaps
from root_cynaps.root_nitrogen import RootNitrogenModel
from root_cynaps.root_water import RootWaterModel
from rhizodep.rhizo_soil import SoilModel
from rhizodep.root_anatomy import RootAnatomy
from Data_enforcer.shoot import ShootModel

from generic_fspm.composite_wrapper import CompositeModel


class Model(CompositeModel):
    """
    Root-CyNAPS model

    Use guideline :
    1. store in a variable Model(g, time_step) to initialize the model, g being an openalea.MTG() object and time_step an time interval in seconds.

    2. print Model.documentation for more information about editable model parameters (optional).

    3. Use Model.scenario(**dict) to pass a set of scenario-specific parameters to the model (optional).

    4. Use Model.run() in a for loop to perform the computations of a time step on the passed MTG File
    """

    def __init__(self, g, time_step: int, **scenario: dict):
        """
        DESCRIPTION
        ----------
        __init__ method of the model. Initializes the thematic modules and link them.

        :param g: the openalea.MTG() instance that will be worked on. It must be representative of a root architecture.
        :param time_step: the resolution time_step of the model in seconds.
        """

        # INIT INDIVIDUAL MODULES
        self.soil = self.load(SoilModel, g, time_step, **scenario)
        self.root_anatomy = self.load(RootAnatomy, g, time_step, **scenario)
        self.root_water = self.load(RootWaterModel, g, time_step, **scenario)
        self.root_nitrogen = self.load(RootNitrogenModel, g, time_step, **scenario)
        self.shoot = ShootModel(g)

        # Order matters here !
        self.models = (self.soil, self.shoot, self.root_water, self.root_nitrogen, self.root_anatomy)

        # LINKING MODULES
        # Get or build translator matrix
        
        try:
            with open(root_cynaps.__path__[0] + "/coupling_translator.yaml", "r") as f:
                translator = yaml.safe_load(f)
        except FileNotFoundError:
            print("NOTE : You will now have to provide information about shared variables between the modules composing this model :\n")
            translator = self.translator_matrix_builder()
            with open(root_cynaps.__path__[0] + "/coupling_translator.yaml", "w") as f:
                yaml.dump(translator, f)

        # Actually link modules together
        self.link_around_mtg(translator)

        # Some initialization must be performed after linking modules
        (m.post_coupling_init() for m in self.models)

    def run(self):
        (m() for m in self.models)
