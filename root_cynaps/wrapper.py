from model_nitrogen import RootNitrogenModel
from model_water import RootWaterModel
from model_soil import HydroMinSoil
from model_topology import RadialTopology
from Data_enforcer.model import ShootModel


class RootCyNAPS:
    """
    Root-CyNAPS model

    Use guideline :
    1. store in a variable RootCyNAPS(g, time_step) to initialize the model, g being an openalea.MTG() object and time_step an time interval in seconds.

    2. print RootCyNAPS.documentation for more information about editable model parameters (optional).

    3. Use RootCyNAPS.scenario(**dict) to pass a set of scenario-specific parameters to the model (optional).

    4. Use RootCyNAPS.run() in a for loop to perform the computations of a time step on the passed MTG Files
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
        # Spatialized root MTG interactions between soil, structure, nitrogen and water
        self.link_mtg(self.root_nitrogen, self.soil, category="soil", same_names=True)
        self.link_mtg(self.root_nitrogen, self.root_topo, category="structure", same_names=True)

        self.link_mtg(self.root_water, self.soil, category="soil", same_names=True)

        self.link_mtg(self.root_water, self.root_topo, category="structure", same_names=True)

        self.link_mtg(self.root_nitrogen, self.root_water, category="water", same_names=True)

        # 1 point collar interactions between shoot CN, root nitrogen and root water
        self.link_mtg(self.root_nitrogen, self.shoot, category="shoot_nitrogen", translator=self.nitrogen_flows,
                           same_names=False)

        self.link_mtg(self.root_water, self.shoot, category="shoot_water", translator=self.water_flows,
                           same_names=False)

        self.root_nitrogen.store_functions_call()
        self.root_water.init_xylem_water()
        self.step = 1

    @property
    def documentation(self):

        """
        Documentation of the RootCyNAPS parameters
        :return: documentation text
        """
        return dict(zip((name, value) for name, value in self.root_nitrogen.__dataclassfields__ if value.metadata["variable_type"] == "state_variable"))

    def scenario(self, **kwargs):
        """
        Method
        """
        for model in self.models:
            for changed_parameter, value in kwargs:
                if changed_parameter in model.__dict__:
                    setattr(model, changed_parameter, value)

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

    def link_mtg(self, receiver, applier, category, translator={}, same_names=True):
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

