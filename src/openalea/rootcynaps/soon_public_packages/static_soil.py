# Public packages
import numpy as np
from dataclasses import dataclass

# Utility packages
from openalea.metafspm.component_factory import *
from openalea.metafspm.component import Model, declare


@dataclass
class SoilModel(Model):
    """
    Empty doc
    """

    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---
    
    # FROM ANATOMY MODEL
    root_exchange_surface: float = declare(default=0., unit="m2", unit_comment="", description="Exchange surface between soil and symplasmic parenchyma.", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")

    # FROM GROWTH MODEL
    length: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example root segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    initial_length: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example root segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")

    # FROM NITROGEN MODEL
    mineralN_uptake: float = declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="", 
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    amino_acids_uptake: float = declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    mineralN_diffusion_from_roots: float =  declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    amino_acids_diffusion_from_roots: float =  declare(default=0., unit="mol.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    mineralN_diffusion_from_xylem: float =  declare(default=0., unit="mol.s-1", unit_comment="of nitrates", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    amino_acids_diffusion_from_xylem: float =  declare(default=0., unit="mol.s-1", unit_comment="of amino_acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
   
    # --- STATE VARIABLES INITIALIZATION ---
    # Intersection with roots
    voxel_neighbor: int = declare(default=None, unit="adim", unit_comment="", description="",
                                                 value_comment="", references="", DOI="",
                                                 min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="descriptor", edit_by="user")
    
    # Temperature
    soil_temperature: float = declare(default=7.8, unit="°C", unit_comment="", description="soil temperature in contact with roots",
                                                 value_comment="Derived from Swinnen et al. 1994 C inputs, estimated from a labelling experiment starting 3rd of March, with average temperature at 7.8 °C", references="Swinnen et al. 1994", DOI="",
                                                 min_value="", max_value="", variable_type="state_variable", by="model_temperature", state_variable_type="intensive", edit_by="user")


    # C&N    
    dissolved_mineral_N: float = declare(default=20e-6, unit="adim", unit_comment="gN per g of dry soil", description="dissolved mineral N massic concentration in soil",
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")

    C_mineralN_soil: float = declare(default=2.2, unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    C_amino_acids_soil: float = declare(default=8.2e-3, unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    microbial_C: float = declare(default=0.2e-3, unit="adim", unit_comment="gC per g of dry soil", description="microbial Carbon massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    microbial_N: float = declare(default=0.03e-3, unit="adim", unit_comment="gN per g of dry soil", description="microbial N massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    

    # All solutes
    Cv_solutes_soil: float = declare(default=32.2 / 10, unit="mol.m-3", unit_comment="mol of  all dissolved mollecules in the soil solution", description="All dissolved mollecules concentration", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")

    # Water related
    water_potential_soil: float = declare(default=-0.1e6, unit="Pa", unit_comment="", description="Mean soil water potential", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    soil_moisture: float = declare(default=0.3, unit="adim", unit_comment="g.g-1", description="Volumetric proportion of water per volume of soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    water_volume: float = declare(default=0.25e-6, unit="m3", unit_comment="", description="Volume of the water in the soil element in contact with a the root segment", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    
    
    # Structure related
    voxel_volume: float = declare(default=1e-6, unit="m3", unit_comment="", description="Volume of the soil element in contact with a the root segment",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    bulk_density: float = declare(default=1.42, unit="g.mL", unit_comment="", description="Volumic density of the dry soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    dry_soil_mass: float = declare(default=1.42, unit="g", unit_comment="", description="dry weight of the considered voxel element", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")

    # --- PARAMETERS ---
    
    ratio_C_per_amino_acid: float = declare(default=6, unit="adim", unit_comment="number of carbon per molecule of amino acid", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # Water-related parameters
    theta_R: float = declare(default=0.0835, unit="adim", unit_comment="m3.m-3", description="Soil retention moisture", 
                                        value_comment="", references="clay loam estimated with Hydrus, bulk density = 1.42", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    theta_S: float = declare(default=0.4383, unit="adim", unit_comment="m3.m-3", description="Soil saturation moisture", 
                                        value_comment="", references="clay loam estimated with Hydrus, bulk density = 1.42", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    water_alpha: float = declare(default=0.0138, unit="cm-3", unit_comment="", description="alpha is the inverse of the air-entry value (or bubbling pressure)", 
                                        value_comment="", references="clay loam estimated with Hydrus, bulk density = 1.42", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    water_n: float = declare(default=1.3945, unit="cm-3", unit_comment="", description="alpha is the inverse of the air-entry value (or bubbling pressure)", 
                                        value_comment="", references="clay loam estimated with Hydrus, bulk density = 1.42", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")


    def __init__(self, time_step_in_seconds, scene_xrange=1., scene_yrange=1., **scenario):
        """
        DESCRIPTION
        -----------
        __init__ method

        :param g: the root MTG
        :param time_step_in_seconds: time step of the simulation (s)
        :param scenario: mapping of existing variable initialization and parameters to superimpose.
        :return:
        """

        self.apply_scenario(**scenario)
        self.initiate_voxel_soil(scene_xrange, scene_yrange)
        self.time_step_in_seconds = time_step_in_seconds
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step_in_seconds, data=self.voxels, compartment="soil")   


    # SERVICE FUNCTIONS

    # Just ressource for now
    def initiate_voxel_soil(self, scene_xrange=1., scene_yrange=1.):
        """
        Note : not tested for now, just computed to support discussions.
        """
        self.voxels = {}

        
        self.planting_depth = 5e-2

        cubic_length = 3e-2
        voxel_width = cubic_length
        voxel_height = cubic_length
        voxel_volume = voxel_height * voxel_width * voxel_width

        self.delta_z = voxel_height
        self.voxels_Z_section_area = voxel_width * voxel_width
        
        self.voxel_number_x = int(scene_xrange / voxel_width) + 1
        actual_voxel_width = scene_xrange / self.voxel_number_x
        self.scene_xrange = scene_xrange

        self.voxel_number_y = int(scene_yrange / voxel_width) + 1
        actual_voxel_length = scene_yrange / self.voxel_number_y
        self.scene_yrange = scene_yrange

        voxel_volume = voxel_height * actual_voxel_width * actual_voxel_length

        scene_zrange = 1.
        self.voxel_number_z = int(scene_zrange / voxel_height) + 1

        # Uncentered, positive grid
        y, z, x = np.indices((self.voxel_number_y, self.voxel_number_z, self.voxel_number_x))
        self.voxels["x1"] = x * actual_voxel_width
        self.voxels["x2"] = self.voxels["x1"] + actual_voxel_width
        self.voxels["y1"] = y * actual_voxel_length
        self.voxels["y2"] = self.voxels["y1"] + actual_voxel_length
        self.voxels["z1"] = z * voxel_height
        self.voxels["z2"] = self.voxels["z1"] + voxel_height

        self.voxel_grid_to_self("voxel_volume", voxel_volume)

        for name in self.state_variables + self.inputs:
            if name != "voxel_volume":
                self.voxel_grid_to_self(name, init_value=getattr(self, name))


    def voxel_grid_to_self(self, name, init_value):
        self.voxels[name] = np.zeros((self.voxel_number_y, self.voxel_number_z, self.voxel_number_x))
        self.voxels[name].fill(init_value)
        #setattr(self, name, self.voxels[name])
    

    def compute_mtg_voxel_neighbors(self, props):

        # necessary to get updated coordinates.
        # if "angle_down" in g.properties().keys():
        #     plot_mtg(g)

        for vid in props["vertex_index"].keys():
            if (vid not in props["voxel_neighbor"].keys()) or (props["voxel_neighbor"][vid] is None) or (props["length"][vid] > props["initial_length"][vid]):
                baricenter = (np.mean((props["x1"][vid], props["x2"][vid])) % self.scene_xrange, # min value is 0
                            np.mean((props["y1"][vid], props["y2"][vid])) % self.scene_yrange, # min value is 0
                            -np.mean((props["z1"][vid], props["z2"][vid])))
                testx1 = self.voxels["x1"] <= baricenter[0]
                testx2 = baricenter[0] <= self.voxels["x2"]
                testy1 = self.voxels["y1"] <= baricenter[1]
                testy2 = baricenter[1] <= self.voxels["y2"]
                testz1 = self.voxels["z1"] <= baricenter[2]
                testz2 = baricenter[2] <= self.voxels["z2"]
                test = testx1 * testx2 * testy1 * testy2 * testz1 * testz2
                try:
                    props["voxel_neighbor"][vid] = [int(v) for v in np.where(test)]
                except:
                    print(" WARNING, issue in computing the voxel neighbor for vid ", vid)
                    props["voxel_neighbor"][vid] = None
        
        return props


    def apply_to_voxel(self, props):
        """
        This function computes the flow perceived by voxels surrounding the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param root_flows: The root flows to be perceived by soil voxels. The underlying assumptions are that only flows, i.e. extensive variables are passed as arguments.
        :return:
        """

        for name in self.inputs:
            self.voxels[name].fill(0)
        
        for vid in props["vertex_index"].keys():
            if props["length"][vid] > 0:
                vy, vz, vx = props["voxel_neighbor"][vid]
                for name in self.inputs:
                    # print(name,  props[name])
                    self.voxels[name][vy][vz][vx] += props[name][vid]


    def get_from_voxel(self, props, soil_outputs):
        """
        This function computes the soil states from voxels perceived by the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param soil_states: The soil states to be perceived by soil voxels. The underlying assumptions are that only intensive extensive variables are passed as arguments.
        :return:
        """
        for vid in props["vertex_index"].keys():
            vy, vz, vx = props["voxel_neighbor"][vid]
            for name in soil_outputs:
                if name != "voxel_neighbor":
                    props[name][vid] = self.voxels[name][vy][vz][vx]
        
        return props


    def pull_available_inputs(self, props):
        # vertices = props["vertex_index"].keys()
        vertices = [vid for vid in props["vertex_index"].keys() if props["living_struct_mass"][vid] > 0]
        
        for input, source_variables in self.pullable_inputs[props["model_name"]].items():
            if input not in props:
                props[input] = {}
            # print(input, source_variables)
            props[input].update({vid: sum([props[variable][vid]*unit_conversion 
                                           for variable, unit_conversion in source_variables.items()]) 
                                 for vid in vertices})
        return props

    
    def __call__(self, queue_plants_to_soil, queues_soil_to_plants, soil_outputs: list=[], *args):

        # We get fluxes and voxel interception from the plant mtgs (If none passed, soil model can be autonomous)
        # Waiting for all plants to put their outputs

        batch = []
        for _ in range(len(queues_soil_to_plants)):
            batch.append(queue_plants_to_soil.get())

        # LINKING MODULES after plant initialization
        keep_props_locally = {}
        for plant_data in batch:
            # Unpacking message
            id = plant_data["plant_id"]
            props = plant_data["data"]
            
            props = self.pull_available_inputs(props)
            props = self.compute_mtg_voxel_neighbors(props)
            self.apply_to_voxel(props)
            keep_props_locally[id] = props

        # Run the soil model
        self.choregrapher(module_family=self.__class__.__name__, *args)

        # Then apply the states to the plants
        for id, props in keep_props_locally.items(): # Not reteived from batch once more since props may be a copy when pulled out of a mp.Queue.get()
            props = self.get_from_voxel(props, soil_outputs=soil_outputs)
            # Update soil properties so that plants can retreive
            queues_soil_to_plants[id].put(props)

    
    # MODEL EQUATIONS
    
    def _soil_moisture(self, water_potential_soil):
        m = 1 - (1/self.water_n)
        return self.theta_R + (self.theta_S - self.theta_R) / (1 + np.abs(self.water_alpha * water_potential_soil)**self.water_n) ** m

    @segmentation
    @state
    def _C_mineralN_soil(self, dissolved_mineral_N, dry_soil_mass, soil_moisture, voxel_volume):
        return dissolved_mineral_N * (dry_soil_mass / (soil_moisture * voxel_volume)) / 14

    #TP@segmentation
    #TP@state
    def _C_amino_acids_soil(self, DOC, dry_soil_mass, soil_moisture, voxel_volume):
        return DOC * (dry_soil_mass * (soil_moisture / voxel_volume)) / 12 / self.ratio_C_per_amino_acid
    
    @postsegmentation
    @state
    def _Cv_solutes_soil(self, C_mineralN_soil, C_amino_acids_soil):
        return C_mineralN_soil + C_amino_acids_soil
    
    @state
    def _water_volume(self, soil_moisture, voxel_volume):
        return soil_moisture * voxel_volume
    
    @state
    def _water_potential_soil(self, voxel_volume, water_volume):
        """
        Water retention curve from van Genuchten 1980
        """
        m = 1 - (1/self.water_n)
        return - (1 / self.water_alpha) * (
                                            ((self.theta_S - self.theta_R) / ((water_volume / voxel_volume) - self.theta_R)) ** (1 / m) - 1 
                                        )** (1 / self.water_n)