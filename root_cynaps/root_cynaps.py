import os
import pickle

from model_nitrogen import RootNitrogenModel
from model_water import RootWaterModel
from model_soil import HydroMinSoil
from model_topology import RadialTopology
from Data_enforcer.model import ShootModel

from wrapper import ModelWrapper


class Model(ModelWrapper):
    """
    Root-CyNAPS model

    Use guideline :
    1. store in a variable Model(g, time_step) to initialize the model, g being an openalea.MTG() object and time_step an time interval in seconds.

    2. print Model.documentation for more information about editable model parameters (optional).

    3. Use Model.scenario(**dict) to pass a set of scenario-specific parameters to the model (optional).

    4. Use Model.run() in a for loop to perform the computations of a time step on the passed MTG Files
    """

    def __init__(self, g, time_step: int):
        """
        DESCRIPTION
        ----------
        __init__ method of the model. Initializes the thematic modules and link them.

        :param g: the openalea.MTG() instance that will be worked on. It must be representative of a root architecture.
        :param time_step: the resolution time_step of the model in seconds.
        """


        # INIT INDIVIDUAL MODULES

        self.soil = HydroMinSoil(g)
        self.root_topo = RadialTopology(g)
        self.root_water = RootWaterModel(g, time_step)
        self.root_nitrogen = RootNitrogenModel(g, time_step, time_step)
        self.shoot = ShootModel(g)
        # Voir initialiser dedans
        self.models = (self.soil, self.root_topo, self.root_water, self.root_nitrogen, self.shoot)

        # LINKING MODULES
        if not os.path.isfile("translator.pckl"):
            self.translator_utility()
        with open("translator.pckl", "rb") as f:
            translator = pickle.load(f)
        self.link_around_mtg(translator)

        # Some initialization must be performed after linking modules
        self.root_nitrogen.store_functions_call()
        self.root_water.init_xylem_water()
        self.step = 1

    def run(self):
        # Update environment boundary conditions
        # Update soil state
        self.soil.update_patches()

        # Compute shoot flows and state balance
        self.shoot.exchanges_and_balance(time=self.step)

        # Compute state variations for water and then nitrogen
        self.root_water.exchanges_and_balance()
        self.root_nitrogen.exchanges_and_balance()

        # Compute root growth from resulting states

        # Update topological surfaces and volumes based on other evolved structural properties
        self.root_topo.update_topology()
        self.step += 1
