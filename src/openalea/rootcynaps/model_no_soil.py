# Components classes
from openalea.rootcynaps import RootAnatomy
from openalea.rootcynaps import RootWaterModel
from openalea.rootcynaps import RootNitrogenModel
from openalea.rootcynaps.soon_public_packages.mtg_structural_init import StaticRootGrowthModel

from multiprocessing.shared_memory import SharedMemory
import numpy as np
from openalea.metafspm.utils import ArrayDict, mtg_to_arraydict
from openalea.fspm.utility.writer.visualize import plot_mtg

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
        # We have to update the coordinates of the new / imported MTG for other model's initialization
        plot_mtg(self.g_root, position=self.coordinates, rotation=self.rotation)
        # NOTE: Requiered here only to initialize some requiered properties on mtg
        self.root_growth = StaticRootGrowthModel(g=scenario["input_mtg"]["root_mtg_file"], time_step=time_step, **root_parameters)
        self.root_anatomy = RootAnatomy(self.g_root, time_step, **root_parameters)
        self.root_water = RootWaterModel(self.g_root, time_step, **root_parameters)
        self.root_nitrogen = RootNitrogenModel(self.g_root, time_step, **root_parameters)
        
        components = (self.root_anatomy, self.root_water, self.root_nitrogen)
        descriptors = []
        for c in components:
            descriptors += c.descriptor

        # NOTE : Important that this type conversion occurs after initiation of the modules 
        # AND BEFORE THE COUPLING FOR ALIASES TO REMAIN UNBROKEN!
        mtg_to_arraydict(self.g_root, ignore=descriptors)

        # LINKING MODULES
        self.declare_data_and_couple_components(root=self.g_root,
                                                translator_path=translator_path,
                                                components=components)
        self.soil_handshake = {v: k for k, v in enumerate(self.plant_side_soil_inputs + self.soil_outputs)}
        
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
        self.root_props["hexose_consumption_by_growth"] = ArrayDict()
        self.root_props["deficit_hexose_root"] = ArrayDict()
        self.root_props["hexose_consumption_by_growth"].update(self.root_props["hexose_consumption_by_growth_rate"])
        self.root_props["deficit_hexose_root"].update(self.root_props["Deficit_hexose_root"]) # 0 init
        # TODO : Transfer to root growth as it is general?
        self.root_props["total_living_struct_mass"][1] = sum(list(self.root_props["living_struct_mass"].values()))
        # Check MTG quality
        for v in self.g_root.vertices(scale=self.g_root.max_scale()):
            n = self.g_root.node(v)
            if n.struct_mass > 0 and not isinstance(n.type, int):
                n.type = self.root_growth.type_Normal_root_after_emergence
                if len(n.children()) > 0:
                    n.label = self.root_growth.label_Segment
                else:
                    n.label = self.root_growth.label_Apex
        
        # Performed in initialization and run to update coordinates
        plot_mtg(self.g_root, position=self.coordinates, rotation=self.rotation)

        self.name = name
        # ROOT PROPERTIES INITIAL PASSING IN MTG
        self.root_props["plant_id"] = name
        self.root_props["model_name"] = self.__class__.__name__
        self.model_name = self.__class__.__name__
        self.carried_components = [component.__class__.__name__ for component in self.components]

        shm = SharedMemory(name=self.name)
        buf = np.ndarray((35,20000), dtype=np.float64, buffer=shm.buf)
        # print(buf)
        for name in self.plant_side_soil_inputs:
            value = self.root_props[name]
            if isinstance(value, ArrayDict):
                buf[self.soil_handshake[name],:len(value)] = value.values_array()
            else:
                print(name, "should be passed")
        
        shm.close()
        self.queue_plants_to_soil.put({"plant_id": self.name, "model_name": self.model_name, "carried_components": self.carried_components, "handshake": self.soil_handshake})


        # Retreive post environments init states
        self.get_environment_boundaries()

        # Send command to environments models to run first
        self.send_plant_status_to_environment()


    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.components, when=self.time)

        # Retrieve soil and light status for plant
        self.get_environment_boundaries()

        # Update mtg coordinates
        plot_mtg(self.g_root, position=self.coordinates, rotation=self.rotation)

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
        shm = SharedMemory(name=self.name)
        buf = np.ndarray((35,20000), dtype=np.float64, buffer=shm.buf)
        vertices = buf[self.soil_handshake["vertex_index"]]
        vertices_mask = vertices >= 1
        for variable_name in self.soil_outputs: # TODO : soil_outputs come from declare_data_and_couple_components, not a good structure to keep
            # print(len(self.root_props[variable_name]))
            if variable_name not in self.root_props.keys(): # Actually used? I am not sure
                self.root_props[variable_name] = ArrayDict()
            
            # self.root_props[variable_name].assign_all(buf[self.soil_handshake[variable_name]][vertices_mask])
            self.root_props[variable_name].scatter(vertices[vertices_mask], buf[self.soil_handshake[variable_name]][vertices_mask])
            
        shm.close()


    def send_plant_status_to_environment(self):
        shm = SharedMemory(name=self.name)
        buf = np.ndarray((35,20000), dtype=np.float64, buffer=shm.buf)
        # print(buf)
        for name in self.plant_side_soil_inputs:
            value = self.root_props[name]
            if isinstance(value, ArrayDict):
                buf[self.soil_handshake[name],:len(value)] = value.values_array()
            else:
                print(name, "should be passed")
        
        shm.close()

        self.queue_plants_to_soil.put({"plant_id": self.name, "model_name": self.model_name, "handshake": self.soil_handshake})