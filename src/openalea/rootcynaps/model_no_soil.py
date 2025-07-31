# Components classes
from openalea.rootcynaps import RootAnatomy
from openalea.rootcynaps import RootWaterModel
from openalea.rootcynaps import RootNitrogenModel
from openalea.rootcynaps.soon_public_packages.mtg_structural_init import StaticRootGrowthModel

# Utilities
from openalea.metafspm.composite_wrapper import CompositeModel
from openalea.metafspm.component_factory import Choregrapher


class RootCyNAPS(CompositeModel):
    """
    Root-BRIDGES model

    Use guideline :
    1. store in a variable Model(g, time_step) to initialize the model, g being an openalea.MTG() object and time_step a time interval in seconds.

    2. print Model.documentation for more information about editable model parameters (optional).

    3. Use Model.scenario(**dict) to pass a set of scenario-specific parameters to the model (optional).

    4. Use Model.run() in a for loop to perform the computations of a time step on the passed MTG File
    """

    def __init__(self, queues_soil_to_plants, queue_plants_to_soil,
                name: str="Plant", time_step: int=3600, coordinates: list=[0, 0, 0], rotation: float=0, translator_path: dict = {}, **scenario):
        """
        DESCRIPTION
        ----------
        __init__ method of the model. Initializes the thematic modules and link them.

        :param g: the openalea.MTG() instance that will be worked on. It must be representative of a root architecture.
        :param time_step: the resolution time_step of the model in seconds.
        """
        # DECLARE GLOBAL SIMULATION TIME STEP, FOR THE CHOREGRAPHER TO KNOW IF IT HAS TO SUBDIVIDE TIME-STEPS
        self.name = name
        self.coordinates = coordinates
        self.rotation = rotation

        Choregrapher().add_simulation_time_step(time_step)
        self.time = 0

        parameters = scenario["parameters"]
        root_parameters = parameters["root_cynaps"]["roots"]
        self.input_tables = scenario["input_tables"]

        # INIT INDIVIDUAL MODULES
        assert len(scenario["input_mtg"]) > 0
        self.g_root = scenario["input_mtg"]["root_mtg_file"]
        # NOTE: Requiered here only to initialize some requiered properties on mtg
        self.root_growth = StaticRootGrowthModel(g=scenario["input_mtg"]["root_mtg_file"], time_step=time_step, **root_parameters)
        self.root_anatomy = RootAnatomy(self.g_root, time_step, **root_parameters)
        self.root_water = RootWaterModel(self.g_root, time_step, **root_parameters)
        self.root_nitrogen = RootNitrogenModel(self.g_root, time_step, **root_parameters)
        
        # LINKING MODULES
        self.declare_data_and_couple_components(root=self.g_root,
                                                translator_path=translator_path,
                                                components=(self.root_anatomy, self.root_water, self.root_nitrogen))
        
        # Specific here TODO remove later
        self.root_water.collar_children = self.root_growth.collar_children
        self.root_water.collar_skip = self.root_growth.collar_skip
        self.root_nitrogen.collar_children = self.root_growth.collar_children
        self.root_nitrogen.collar_skip = self.root_growth.collar_skip

        # Provide signature for the MTG
        # Retreive the queues to communicate with environment models
        self.queues_soil_to_plants=queues_soil_to_plants
        self.queue_plants_to_soil=queue_plants_to_soil

        # Get properties from each MTG
        self.root_props = self.g_root.properties()
        # Note specific property name adaptation to work from Frederic's RhizoDep outputs
        self.root_props["hexose_consumption_by_growth"] = {}
        self.root_props["hexose_consumption_by_growth"].update(self.root_props["hexose_consumption_by_growth_rate"])
        # TODO : Transfer to root growth as it is general?
        self.root_props["total_living_struct_mass"][1] = sum(list(self.root_props["living_struct_mass"].values()))
        # Check MTG quality
        for v in self.g_root.vertices(scale=self.g_root.max_scale()):
            n = self.g_root.node(v)
            if n.struct_mass > 0 and not isinstance(n.type, str):
                n.type = 'Normal_root_after_emergence'
                if len(n.children()) > 0:
                    n.label = 'Segment'
                else:
                    n.label = 'Apex'
        
        # Performed in initialization and run to update coordinates
        print("WARNING, coordinates updating has been manually commented to use input MTGs")
        # plot_mtg(self.g_root, position=self.coordinates, rotation=self.rotation)

        self.name = name
        # ROOT PROPERTIES INITIAL PASSING IN MTG
        self.root_props["plant_id"] = name
        self.root_props["model_name"] = self.__class__.__name__
        self.root_props["carried_components"] = [component.__class__.__name__ for component in self.components]
        self.queue_plants_to_soil.put({"plant_id": self.name, "data": self.root_props})

        # Retreive post environments init states
        self.get_environment_boundaries()

        # Send command to environments models to run first
        self.send_plant_status_to_environment()


    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.components, when=self.time)

        # Retrieve soil and light status for plant
        self.get_environment_boundaries()

        # Update topological surfaces and volumes based on other evolved structural properties
        self.root_anatomy()

        # Compute state variations for water and then carbon and nitrogen
        self.root_water()
        self.root_nitrogen()

        # Send plant status to soil and light models
        self.send_plant_status_to_environment()

        self.time += 1


    def get_environment_boundaries(self):
        # Wait for results from both soil and light model before begining
        soil_boundary_props = self.queues_soil_to_plants[self.name].get()

        # NOTE : here you have to perform a per-variable update otherwise dynamic links are broken
        for variable_name in self.soil_outputs + ["voxel_neighbor"]: # TODO : soil_outputs come from declare_data_and_couple_components, not a good structure to keep
            if variable_name not in self.root_props.keys():
                self.root_props[variable_name] = {}
            
            self.root_props[variable_name].update(soil_boundary_props[variable_name])


    def send_plant_status_to_environment(self):
        self.queue_plants_to_soil.put({"plant_id": self.name, "data": self.root_props})