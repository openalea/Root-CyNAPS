"""
root_cynaps.nitrogen
_________________
This is the main nitrogen cycle module for roots

Documentation and features
__________________________

Main functions
______________
Classes' names represent accounted hypothesis in the progressive development of the model.
Methods' names are systematic through all class for ease of use :

TODO : report functions descriptions
CommonNitrogenModel
    init_N()
    transport_N()
    metabolism_N()
    update_N()

DiscreteVessels(CommonNitrogenModel)


"""
# Imports
import numpy as np
from dataclasses import dataclass, asdict
import multiprocessing as mp
from multiprocessing import shared_memory
import inspect as ins
from functools import partial


# Dataclass for initialisation and parametrization.
# For readability's sake, only units are displayed. See functions' documentation for descriptions.

# Properties' initial values
@dataclass
class InitNitrogen:
    # time resolution
    sub_time_step: int = 3600  # (second) MUST be a multiple of base time_step
    # Pools initial size
    Nm: float = 1e-4  # mol N.g-1
    AA: float = 9e-4  # mol AA.g-1
    struct_protein: float = 0  # mol prot struct.g-1
    storage_protein: float = 0  # mol prot stor.g-1
    xylem_Nm: float = 1e-4  # mol N.g-1
    xylem_AA: float = 1e-4  # mol AA.g-1
    # Transport processes
    import_Nm: float = 0  # mol N.s-1
    import_AA: float = 0  # mol AA.s-1
    export_Nm: float = 0  # mol N.s-1
    export_AA: float = 0  # mol AA.s-1
    diffusion_Nm_soil: float = 0  # mol N.s-1
    diffusion_Nm_xylem: float = 0  # mol N.s-1
    diffusion_Nm_soil_xylem: float = 0  # mol N.s-1
    diffusion_AA_soil: float = 0  # mol AA.s-1
    diffusion_AA_phloem: float = 0  # mol AA.s-1
    diffusion_AA_soil_xylem: float = 0  # mol AA.s-1
    # Metabolic processes
    AA_synthesis: float = 0  # mol AA.s-1
    struct_synthesis: float = 0  # mol struct.s-1
    storage_synthesis: float = 0  # mol stor.s-1
    AA_catabolism: float = 0  # mol AA.s-1
    storage_catabolism: float = 0  # mol stor.s-1
    phloem_total_AA: float = 9e-4  # mol AA.g-1
    total_cytokinins: float = 100  # Artif UA.g-1
    cytokinin_synthesis: float = 0  # UA cytokinin.s-1
    xylem_struct_mass: float = 1e-6  # g
    displaced_Nm_in: float = 0  # mol Nm.time_step-1
    displaced_Nm_out: float = 0  # mol Nm.time_step-1
    displaced_AA_in: float = 0  # mol Nm.time_step-1
    displaced_AA_out: float = 0  # mol Nm.time_step-1
    cumulated_radial_exchanges_Nm: float = 0  # mol Nm.time_step-1
    cumulated_radial_exchanges_AA: float = 0  # mol AA.time_step-1
    phloem_struct_mass: float = 5e-7  # g


# Parameters' default value
@dataclass
class ProcessNitrogen:
    # N TRANSPORT PROCESSES

    # kinetic parameters
    vmax_Nm_root: float = 1e-6  # mol N.s-1.m-2
    vmax_Nm_xylem: float = 1e-6  # mol N.s-1.m-2
    Km_Nm_root_LATS: float = 1e-1  # mol N.m-3 Changed to increase diminution
    Km_Nm_root_HATS: float = 1e-6  # mol N.m-3
    begin_N_regulation: float = 1e1  # Artif mol N.g-1 changed so that import_Nm variation may occur in Nm variation range
    span_N_regulation: float = 2e-4  # mol N.g-1 range corresponding to observed variation range within segment
    Km_Nm_xylem: float = 8e-5  # mol N.g-1
    vmax_AA_root: float = 1e-7  # mol AA.s-1.m-2
    Km_AA_root: float = 1e-1  # mol AA.m-3
    vmax_AA_xylem: float = 1e-7  # mol AA.s-1.m-2
    Km_AA_xylem: float = 8e-5  # mol AA.g-1
    diffusion_soil: float = 1e-11  # Artif g.m-2.s-1 while there is no soil model balance
    diffusion_xylem: float = 0  # 1e-6   # Artif g.m-2.s-1 It was noticed it only contributed to xylem loading
    diffusion_phloem: float = 1e-5  # Artif *1e-1 g.m-2.s-1 more realistic ranges
    diffusion_apoplasm: float = 2.5e-10  # Artif. g.m-2.s-1 while there is no soil model balance

    # metabolism-related parameters
    transport_C_regulation: float = 7e-3  # mol.g-1

    # N METABOLISM PROCESSES
    # TODO : introduce nitrogen fixation

    # kinetic parameters
    smax_AA: float = 1e-5  # Artif mol.s-1.g-1 DW
    Km_Nm_AA: float = 3e-6  # mol.g-1 DW
    Km_C_AA: float = 350e-6  # mol.g-1 DW
    smax_struct: float = 1e-9  # mol.s-1.g-1 DW
    Km_AA_struct: float = 250e-6  # mol.g-1 DW
    smax_stor: float = 0  # Artif 1e-9  # mol.s-1.g-1 DW 0 for wheat
    Km_AA_stor: float = 250e-6  # mol.g-1 DW
    cmax_stor: float = 1e-9  # mol.s-1.g-1 DW
    Km_stor_catab: float = 250e-6  # mol.g-1 DW
    cmax_AA: float = 0  # Artif 5e-9    # mol.s-1.g-1 DW for now not relevant as it doesn't contribute to C_hexose_root balance
    Km_AA_catab: float = 2.5e-6  # mol.g-1 DW
    storage_C_regulation: float = 3e1  # mol.g-1 Changed to avoid reaching Vmax with slight decrease in hexose content

    # HORMONES METABOLISM PROCESSES

    # kinetic parameters
    smax_cytok: float = 9e-4  # UA.g DW-1.s-1
    Km_C_cytok: float = 1.2e-3
    Km_N_cytok: float = 5e-5


@dataclass
class UpdateNitrogen:
    r_Nm_AA: float = 1.4
    r_AA_struct: float = 65
    r_AA_stor: float = 65
    xylem_cross_area_ratio: float = 0.84 * (0.36 ** 2)  # (adim) apoplasmic cross-section area ratio * stele radius ratio^2
    phloem_cross_area_ratio: float = 0.15 * (0.36 ** 2)  # (adim) phloem cross-section area ratio * stele radius ratio^2


class RootNitrogenModel:
    def __init__(self, g, time_step, sub_time_step, Nm, AA, struct_protein, storage_protein, import_Nm, import_AA,
                 export_Nm, export_AA, diffusion_Nm_soil, diffusion_Nm_xylem, diffusion_Nm_soil_xylem,
                 diffusion_AA_soil, diffusion_AA_phloem, diffusion_AA_soil_xylem, AA_synthesis, struct_synthesis,
                 storage_synthesis, AA_catabolism, storage_catabolism, phloem_total_AA, total_cytokinins,
                 cytokinin_synthesis, xylem_Nm, xylem_AA, xylem_struct_mass, displaced_Nm_in, displaced_Nm_out,
                 displaced_AA_in, displaced_AA_out, cumulated_radial_exchanges_Nm, cumulated_radial_exchanges_AA,
                 phloem_struct_mass):

        """
        Description
        Initialization of nitrogen-related variables

        Parameters
        :param g: MTG
        :param Nm: Local mineral nitrogen volumic concentration (mol.m-3)
        :param import_Nm: Local mineral nitrogen influx from soil (mol.s-1)
        :param export_Nm: Local mineral nitrogen loading to xylem (mol.s-1)
        :param diffusion_Nm_phloem: Local mineral nitrogen diffusion between cortex and phloem (mol.s-1)
        :param xylem_Nm: Global xylem mineral nitrogen volumic concentration (mol.m-3)
        :param xylem_volume: Global xylem vessel volume (m3)
        :param phloem_Nm: Global phloem mineral nitrogen volumic concentration (mol.m-3)
        :param phloem_volume: Global phloem vessel volume (m3)
        :param Nm_root_shoot_xylem: Mineral nitrogen transport to shoot from root xylem (mol.s-1)
        :param Nm_root_shoot_phloem: Mineral nitrogen transport from shoot to root phloem (mol.s-1)

        Hypothesis
        H1 :
        H2 :
        """

        self.g = g
        self.props = self.g.properties()
        self.time_step = time_step
        self.sub_time_step = sub_time_step

        # New properties' creation in MTG
        self.keywords = dict(
            Nm=Nm,
            AA=AA,
            struct_protein=struct_protein,
            storage_protein=storage_protein,
            import_Nm=import_Nm,
            import_AA=import_AA,
            export_Nm=export_Nm,
            export_AA=export_AA,
            diffusion_Nm_soil=diffusion_Nm_soil,
            diffusion_Nm_xylem=diffusion_Nm_xylem,
            diffusion_Nm_soil_xylem=diffusion_Nm_soil_xylem,
            diffusion_AA_soil=diffusion_AA_soil,
            diffusion_AA_phloem=diffusion_AA_phloem,
            diffusion_AA_soil_xylem=diffusion_AA_soil_xylem,
            AA_synthesis=AA_synthesis,
            struct_synthesis=struct_synthesis,
            storage_synthesis=storage_synthesis,
            AA_catabolism=AA_catabolism,
            storage_catabolism=storage_catabolism,
            xylem_Nm=xylem_Nm,
            xylem_AA=xylem_AA,
            xylem_struct_mass=xylem_struct_mass,
            displaced_Nm_in=displaced_Nm_in,
            displaced_Nm_out=displaced_Nm_out,
            displaced_AA_in=displaced_AA_in,
            displaced_AA_out=displaced_AA_out,
            cumulated_radial_exchanges_Nm=cumulated_radial_exchanges_Nm,
            cumulated_radial_exchanges_AA=cumulated_radial_exchanges_AA,
            phloem_struct_mass=phloem_struct_mass
        )

        for name in self.keywords:
            self.props.setdefault(name, {})

        # vertices storage for future calls in for loops
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        for vid in self.vertices:
            for name, value in self.keywords.items():
                # Effectively creates the new property
                self.props[name][vid] = value

        # Accessing properties once, pointing to g for further modifications
        self.states = """
                        Nm
                        AA
                        struct_protein
                        storage_protein
                        volume
                        import_Nm
                        import_AA
                        export_Nm
                        export_AA
                        diffusion_Nm_soil
                        diffusion_Nm_xylem
                        diffusion_Nm_soil_xylem
                        diffusion_AA_soil
                        diffusion_AA_phloem
                        diffusion_AA_soil_xylem
                        AA_synthesis
                        struct_synthesis
                        storage_synthesis
                        AA_catabolism
                        storage_catabolism
                        xylem_Nm
                        xylem_AA
                        xylem_struct_mass
                        displaced_Nm_in
                        displaced_Nm_out
                        displaced_AA_in
                        displaced_AA_out
                        cumulated_radial_exchanges_Nm
                        cumulated_radial_exchanges_AA
                        phloem_struct_mass
                        length
                        radius
                        struct_mass
                        C_hexose_root
                        C_hexose_reserve
                        struct_mass_produced
                        living_root_hairs_external_surface
                        thermal_time_since_emergence
                        """.split()

        # Declare MTG properties in self
        for name in self.states:
            setattr(self, name, self.props[name])

        # Repeat the same process for total root system properties

        # Creating variables for
        self.totals_keywords = dict(total_Nm=0,
                                    total_AA=0,
                                    total_hexose=0,
                                    total_cytokinins=total_cytokinins,
                                    total_struct_mass=sum(self.struct_mass.values()),
                                    total_xylem_Nm=0,
                                    total_xylem_AA=0,
                                    total_phloem_AA=phloem_total_AA,
                                    Nm_root_shoot_xylem=0,
                                    AA_root_shoot_xylem=0,
                                    total_AA_rhizodeposition=0,
                                    cytokinin_synthesis=cytokinin_synthesis
                                    )

        for name, value in self.totals_keywords.items():
            self.props.setdefault(name, {})
            self.props[name][1] = value

        # Accessing properties once, pointing to g for further modifications
        self.totals_states = """
                                    total_Nm
                                    total_AA
                                    total_hexose
                                    total_cytokinins
                                    total_struct_mass
                                    total_xylem_Nm
                                    total_xylem_AA
                                    total_phloem_AA
                                    Nm_root_shoot_xylem
                                    AA_root_shoot_xylem
                                    total_AA_rhizodeposition
                                    cytokinin_synthesis
                                    """.split()

        # Declare MTG properties in self
        for name in self.totals_states:
            setattr(self, name, self.props[name])

        # Declare to outside modules which variables are needed
        self.inputs = {
            # Common
            "soil": [
                "soil_Nm",
                "soil_AA"
            ],
            "structure": [
                "root_exchange_surface",
                "stele_exchange_surface",
                "phloem_exchange_surface",
                "apoplasmic_stele"
            ],
            "carbon": [
            ],
            "shoot_nitrogen": [
                "AA_root_shoot_phloem",
                "cytokinins_root_shoot_xylem"
            ],
            "water": [
                "xylem_water",
                "axial_export_water_up",
                "axial_import_water_down"
            ]
        }

    def store_functions_call(self):
        # Storing function calls
        #
        # Local and plant scale processes...
        self.process_param = asdict(ProcessNitrogen())
        self.process_methods = [partial(getattr(self, func), **self.process_param)
                                for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'process' in func)]
        self.process_args = [[partial(self.get_up_to_date, arg) for arg in ins.getfullargspec(getattr(self, func))[0] if arg != "self"]
                                for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'process' in func)]
        self.process_names = [func[8:] for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'process' in func)]

        # Local and plant scale update...
        self.update_param = asdict(UpdateNitrogen())
        self.update_methods = [partial(getattr(self, func), **self.update_param)
                                for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'update' in func)]
        self.update_args = [[partial(self.get_up_to_date, arg) for arg in ins.getfullargspec(getattr(self, func))[0] if arg != "self"]
                             for func in dir(self) if
                             (callable(getattr(self, func)) and '__' not in func and 'update' in func)]
        self.update_names = [func[7:] for func in dir(self) if
                              (callable(getattr(self, func)) and '__' not in func and 'update' in func)]
        print(self.update_args)


        self.plant_scale_update_methods = [partial(getattr(self, func), **self.update_param)
                                              for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'actualize_total' in func)]
        self.plant_scale_update_args = [[partial(self.get_up_to_date, arg) for arg in ins.getfullargspec(getattr(self, func))[0] if arg != "self"]
                             for func in dir(self) if
                             (callable(getattr(self, func)) and '__' not in func and 'actualize_total' in func)]
        self.plant_scale_update_names = [func[10:] for func in dir(self) if
                              (callable(getattr(self, func)) and '__' not in func and 'actualize_total' in func)]

        # TODO : Chunk with map function across vid or processes chunks
        num_processes = mp.cpu_count()
        self.p = mp.Pool(num_processes)

    def exchanges_and_balance(self, parallel=False):
        """
        Description
        ___________
        Model time-step processes and balance for nitrogen to be called by simulation files.

        """

        self.add_properties_to_new_segments()
        self.initialize_cumulative()

        # For each sub_time_step
        for k in range(int(self.time_step/self.sub_time_step)):
            if parallel:
                chunk_size = 1000
                vertices_chunks = [self.vertices[i:i + chunk_size] if i + chunk_size < len(self.vertices) else self.vertices[i:] for i in range(0, len(self.vertices), chunk_size)]
                result = self.p.apply_async(self.prc_resolution, vertices_chunks)
                result.wait()
                result = self.p.apply_async(self.upd_resolution, vertices_chunks)
                result.wait()
                # TODO Share objects between processes to avoid copies slowing down the computation and modify self
                # Might be easier with xarray around a shared dataset with Dask implementation.
            else:
                self.props.update(self.prc_resolution())
                self.resolution_over_vertices(self.vertices, fncs=[self.axial_transport_N])
                self.props.update(self.upd_resolution())
            # Perform global properties' update
            self.props.update(self.tot_upd_resolution())

    def prc_resolution(self):
        return dict(zip([name for name in self.process_names], map(self.dict_mapper, *(self.process_methods, self.process_args))))

    def upd_resolution(self):
        # TODO, ask if the struct_mass > 0 should be operated here and not in each mapped function
        return dict(zip([name for name in self.update_names], map(self.dict_mapper, *(self.update_methods, self.update_args))))

    def tot_upd_resolution(self):
        return dict(zip([name for name in self.plant_scale_update_names], map(self.dict_no_mapping, *(self.plant_scale_update_methods, self.plant_scale_update_args))))

    def dict_mapper(self, fcn, args):
        return dict(zip(args[0](), map(fcn, *(d().values() for d in args))))

    def dict_no_mapping(self, fcn, args):
        return {1: fcn(*(d() for d in args))}

    def get_up_to_date(self, prop):
        return getattr(self, prop)

    def resolution_over_vertices(self, chunk, fncs, **kwargs):
        for vid in chunk:
            if self.struct_mass[vid] > 0:
                for method in fncs:
                    method(v=vid, **kwargs)

    def add_properties_to_new_segments(self):
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        for vid in self.vertices:
            if vid not in list(self.Nm.keys()):
                for prop in list(self.keywords.keys()):
                    getattr(self, prop)[vid] = self.keywords[prop]
        # WARNING? OPTIONAL AND TO REMOVE WHEN NO SIMULATION FROM FILE
        for name in self.states:
            setattr(self, name, self.g.properties()[name])

    def initialize_cumulative(self):
        # Reinitialize for the sum of the next loop
        self.Nm_root_shoot_xylem[1] = 0
        self.AA_root_shoot_xylem[1] = 0
        for vid in self.vertices:
            # Cumulative flows are reinitialized
            self.cumulated_radial_exchanges_Nm[vid] = 0
            self.cumulated_radial_exchanges_AA[vid] = 0
            self.displaced_Nm_out[vid] = 0
            self.displaced_AA_out[vid] = 0
            self.displaced_Nm_in[vid] = 0
            self.displaced_AA_in[vid] = 0

    # NITROGEN PROCESSES

    # RADIAL TRANSPORT PROCESSES
    # MINERAL NITROGEN TRANSPORT
    def process_import_Nm(self, Nm, soil_Nm, root_exchange_surface, C_hexose_root, living_root_hairs_external_surface, **kwargs):
        """
                Description
                ___________
                Nitrogen transport between local soil, local root segment and global vessels (xylem and phloem).

                Parameters
                __________
                :param Km_Nm_root: Active transport from soil Km parameter (mol.m-3)
                :param vmax_Nm_emergence: Surfacic maximal active transport rate in roots (mol.m-2.s-1)
                :param Km_Nm_xylem: Active transport from root Km parameter (mol.m-3)
                :param diffusion_phloem: Mineral nitrogen diffusion parameter (m.s-1)
                :param transport_C_regulation: Km coefficient for the nitrogen active transport regulation function
                by root C (mol.g-1) (?)
                by root mineral nitrogen (mol.m-3)

                Hypothesis
                __________
                H1: We summarize radial active transport controls (transporter density, affinity regulated with genetics
                or environmental control, etc) as one mean transporter following Michaelis Menten's model.

                H2: We can summarize apoplastic and symplastic radial transport through one radial transport.
                Differentiation with epidermis conductance loss, root hair density, aerenchyma, etc, is supposed to impact Vmax.

                H3: We declare similar kinetic parameters for soil-root and root-xylem active transport (exept for concentration conflict)
                """
        # We define mineral nitrogen active uptake from soil
        precision = 0.99
        Km_Nm_root = (kwargs["Km_Nm_root_LATS"] - kwargs["Km_Nm_root_HATS"]) / (
                1 + (precision / ((1 - precision) * np.exp(-kwargs["begin_N_regulation"]))
                     * np.exp(-Nm / kwargs["span_N_regulation"]))
        ) + kwargs["Km_Nm_root_HATS"]
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        return ((soil_Nm * kwargs["vmax_Nm_root"] / (soil_Nm + Km_Nm_root))
                * (root_exchange_surface + living_root_hairs_external_surface)
                * (C_hexose_root / (C_hexose_root + kwargs["transport_C_regulation"])))

    def process_diffusion_Nm_soil(self, Nm, soil_Nm, root_exchange_surface, living_root_hairs_external_surface, **kwargs):
        # Passive radial diffusion between soil and cortex.
        # It happens only through root segment external surface.
        # We summarize apoplasm-soil and cortex-soil diffusion in 1 flow.
        return (kwargs["diffusion_soil"] * (Nm * 10e5 - soil_Nm) * (
                root_exchange_surface + living_root_hairs_external_surface))

    def process_export_Nm(self, Nm, stele_exchange_surface, C_hexose_root, **kwargs):
        # We define active export to xylem from root segment
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        return ((Nm * kwargs["vmax_Nm_xylem"]) / (Nm + kwargs["Km_Nm_xylem"])) * stele_exchange_surface * (
                C_hexose_root / (C_hexose_root + kwargs["transport_C_regulation"]))

    def process_diffusion_Nm_xylem(self, xylem_Nm, Nm, stele_exchange_surface, **kwargs):
        # Passive radial diffusion between xylem and cortex through plasmalema
        return kwargs["diffusion_xylem"] * (xylem_Nm - Nm) * stele_exchange_surface

    def process_diffusion_Nm_soil_xylem(self, soil_Nm, xylem_Nm, radius, length, apoplasmic_stele, **kwargs):
        # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
        # Here, surface is not really representative of a structure as everything is apoplasmic
        return kwargs["diffusion_apoplasm"] * (
                soil_Nm - xylem_Nm * 10e5) * 2 * np.pi * radius * length * apoplasmic_stele

    # AMINO ACID TRANSPORT
    def process_import_AA(self, soil_AA, root_exchange_surface, living_root_hairs_external_surface, C_hexose_root, **kwargs):
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        return ((soil_AA * kwargs["vmax_AA_root"] / (soil_AA + kwargs["Km_AA_root"]))
                * (root_exchange_surface + living_root_hairs_external_surface)
                * (C_hexose_root / (C_hexose_root + kwargs["transport_C_regulation"])))

    def process_diffusion_AA_soil(self, AA, soil_AA, root_exchange_surface, living_root_hairs_external_surface, **kwargs):
        # We define amino acid passive diffusion to soil
        return (kwargs["diffusion_soil"] * (AA * 10e5 - soil_AA)
                * (root_exchange_surface + living_root_hairs_external_surface))

    def process_export_AA(self, AA, stele_exchange_surface, C_hexose_root, **kwargs):
        # We define active export to xylem from root segment
        # Km is defined as a constant here
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        return ((AA * kwargs["vmax_AA_xylem"] / (AA + kwargs["Km_AA_xylem"]))
                * stele_exchange_surface * (C_hexose_root / (
                        C_hexose_root + kwargs["transport_C_regulation"])))

    def process_diffusion_AA_soil_xylem(self, soil_AA, xylem_AA, radius, length, apoplasmic_stele, **kwargs):
        # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
        return (kwargs["diffusion_apoplasm"] * (soil_AA - xylem_AA * 10e5)
                * 2 * np.pi * radius * length * apoplasmic_stele)

    def process_diffusion_AA_phloem(self, AA, phloem_exchange_surface, **kwargs):
        # Passive radial diffusion between phloem and cortex through plasmodesmata
        # TODO : Change diffusive flow to enable realistic ranges, now, unloading is limited by a ping pong bug related to diffusion
        # TODO : resolve exception when mapping has to deal with plant scale properties AND local ones
        return (kwargs["diffusion_phloem"] * (self.total_phloem_AA[1] - AA)
                * phloem_exchange_surface)

    # AXIAL TRANSPORT PROCESSES
    def axial_transport_N(self, v, **kwargs):
        """
                Description
                ___________

        """
        # TODO : probably collar children have to be reintroduced for good neighbor management. (from model_water)
        #  But if null water content proprely passes information between collar and it's children,
        #  it may be already working well

        # AXIAL TRANSPORT

        # If this is only an out flow to up parents
        if self.axial_export_water_up[v] > 0:
            # Turnover defines a dilution factor of radial transport processes over the axially transported
            # water column
            turnover = self.axial_export_water_up[v] / self.xylem_water[v]
            if turnover <= 1:
                # Transport only affects considered segment
                self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * self.sub_time_step
                self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * self.sub_time_step
                # Exported matter corresponds to the exported water proportion
                self.displaced_Nm_out[v] = turnover * self.xylem_Nm[v] * self.xylem_struct_mass[v]
                self.displaced_AA_out[v] = turnover * self.xylem_AA[v] * self.xylem_struct_mass[v]
                up_parent = self.g.parent(v)
                # If this is collar, this flow is exported
                if up_parent == None:
                    self.Nm_root_shoot_xylem[1] += self.displaced_Nm_out[v]
                    self.AA_root_shoot_xylem[1] += self.displaced_AA_out[v]
                else:
                    # The immediate parent receives this flow
                    self.displaced_Nm_in[up_parent] += self.displaced_Nm_out[v]
                    self.displaced_AA_in[up_parent] += self.displaced_AA_out[v]
            else:
                #print("Uturnover >1")
                # Exported matter corresponds to the whole segment's water content
                self.displaced_Nm_out[v] = self.xylem_Nm[v] * self.xylem_struct_mass[v]
                self.displaced_AA_out[v] = self.xylem_AA[v] * self.xylem_struct_mass[v]
                # Transport affects a chain of parents
                water_exchange_time = self.sub_time_step / turnover
                # Loading of the current vertex into the current vertex's xylem
                self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time
                self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time

                exported_water = self.axial_export_water_up[v]
                child = v
                # Loading of the current vertex into the vertices who have received water from it
                while exported_water > 0:
                    # We remove the amount of water which has already received loading in previous loop
                    exported_water -= self.xylem_water[child]
                    up_parent = self.g.parent(child)
                    # If we reached collar, this amount is being exported
                    if up_parent == None:
                        self.Nm_root_shoot_xylem[1] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * exported_water / self.xylem_water[v]
                        self.AA_root_shoot_xylem[1] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * exported_water / self.xylem_water[v]
                        # If all water content of initial segment is exported through collar
                        if exported_water > self.xylem_water[v]:
                            self.Nm_root_shoot_xylem[1] += self.displaced_Nm_out[v]
                            self.AA_root_shoot_xylem[1] += self.displaced_AA_out[v]
                        else:
                            parent_proportion = exported_water / self.xylem_water[v]
                            self.Nm_root_shoot_xylem[1] += self.displaced_Nm_out[v] * parent_proportion
                            self.AA_root_shoot_xylem[1] += self.displaced_AA_out[v] * parent_proportion
                            self.displaced_Nm_in[child] += self.displaced_Nm_out[v] * (1 - parent_proportion)
                            self.displaced_AA_in[child] += self.displaced_AA_out[v] * (1 - parent_proportion)
                        # Break the loop
                        exported_water = 0
                    else:
                        # If the considered parent have been completly filled with water from the child
                        if exported_water - self.xylem_water[up_parent] > 0:
                            # The exposition time is longer if the water content of the target neighbour is more important.
                            self.cumulated_radial_exchanges_Nm[up_parent] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * self.xylem_water[up_parent] / self.xylem_water[v]
                            self.cumulated_radial_exchanges_AA[up_parent] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * self.xylem_water[up_parent] / self.xylem_water[v]
                        # If it's only partial, we account only for the exceeding amount
                        else:
                            self.cumulated_radial_exchanges_Nm[up_parent] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * exported_water / self.xylem_water[v]
                            self.cumulated_radial_exchanges_AA[up_parent] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * exported_water / self.xylem_water[v]
                            # If all water content of initial segment is exported to the considered grandparent
                            if exported_water > self.xylem_water[v]:
                                self.displaced_Nm_in[up_parent] += self.displaced_Nm_out[v]
                                self.displaced_AA_in[up_parent] += self.displaced_AA_out[v]
                            else:
                                # Displaced matter is shared between child and its parent
                                parent_proportion = exported_water / self.xylem_water[v]
                                self.displaced_Nm_in[up_parent] += self.displaced_Nm_out[v] * parent_proportion
                                self.displaced_AA_in[up_parent] += self.displaced_AA_out[v] * parent_proportion
                                self.displaced_Nm_in[child] += self.displaced_Nm_out[v] * (1 - parent_proportion)
                                self.displaced_AA_in[child] += self.displaced_AA_out[v] * (1 - parent_proportion)
                            # Break the loop
                            exported_water = 0
                        child = up_parent

        # If this is only a out flow to down children
        if self.axial_import_water_down[v] < 0:
            # Turnover defines a dilution factor of radial transport processes over the axially transported
            # water column
            turnover = - self.axial_import_water_down[v] / self.xylem_water[v]
            if turnover <= 1:
                # Transport only affects considered segment
                self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * self.sub_time_step
                self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * self.sub_time_step
                # Exported matter corresponds to the exported water proportion
                self.displaced_Nm_out[v] = turnover * self.xylem_Nm[v] * self.xylem_struct_mass[v]
                self.displaced_AA_out[v] = turnover * self.xylem_AA[v] * self.xylem_struct_mass[v]
                down_children = [k for k in self.g.children(v) if self.struct_mass[k] > 0]
                # The immediate children receive this flow
                radius_sum = sum([self.radius[k] for k in down_children])
                children_radius_prop = [self.radius[k] / radius_sum for k in down_children]
                for ch in range(len(down_children)):
                    self.displaced_Nm_in[down_children[ch]] += self.displaced_Nm_out[v] * children_radius_prop[ch]
                    self.displaced_AA_in[down_children[ch]] += self.displaced_AA_out[v] * children_radius_prop[ch]

            else:
                # Transport affects several segments, and we verified it often happens under high transpiration
                # Exported matter corresponds to the whole segment's water content
                self.displaced_Nm_out[v] = self.xylem_Nm[v] * self.xylem_struct_mass[v]
                self.displaced_AA_out[v] = self.xylem_AA[v] * self.xylem_struct_mass[v]
                # Transport affects a chain of children
                water_exchange_time = self.sub_time_step / turnover
                # Loading of the current vertex into the current vertex's xylem
                self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time
                self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time

                parent = [v]
                # We initialize a list tracking water repartition among down axes
                axis_proportion = [1.0]
                # We remove the amount of water which has already been received
                exported_water = [-self.axial_import_water_down[v] - self.xylem_water[v]]
                # Loading of the current vertex into the vertices who have received water from it
                while True in [k > 0 for k in exported_water]:
                    children_list = []
                    children_exported_water = []
                    for p in range(len(parent)):
                        if exported_water[p] > 0:
                            down_children = [k for k in self.g.children(parent[p]) if self.struct_mass[k] > 0]
                            # if the parent is an apex and water has been exported from it,
                            # it means that the apex concentrates the associated carried and loaded nitrogen matter
                            if len(down_children) == 0:
                                # this water amount has also been subject to loading
                                self.cumulated_radial_exchanges_Nm[parent[p]] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * exported_water[p] / self.xylem_water[v]
                                self.cumulated_radial_exchanges_AA[parent[p]] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * exported_water[p] / self.xylem_water[v]
                                # if the translated nitrogen matter has completely ended up in the apex
                                if exported_water[p] + self.xylem_water[parent[p]] > self.xylem_water[v] * axis_proportion[p]:
                                    self.displaced_Nm_in[parent[p]] += self.displaced_Nm_out[v] * axis_proportion[p]
                                    self.displaced_AA_in[parent[p]] += self.displaced_AA_out[v] * axis_proportion[p]
                                # else it is shared with grandparent
                                else:
                                    grandparent = self.g.parent(parent[p])
                                    parent_proportion = (exported_water[p] + self.xylem_water[parent[p]]) / (self.xylem_water[v] * axis_proportion[p])
                                    self.displaced_Nm_in[parent[p]] += self.displaced_Nm_out[v] * axis_proportion[p] * parent_proportion
                                    self.displaced_AA_in[parent[p]] += self.displaced_AA_out[v] * axis_proportion[p] * parent_proportion
                                    self.displaced_Nm_in[grandparent] += self.displaced_Nm_out[v] * axis_proportion[p] * (1 - parent_proportion)
                                    self.displaced_AA_in[grandparent] += self.displaced_AA_out[v] * axis_proportion[p] * (1 - parent_proportion)

                            # If there is only 1 child (root line)
                            elif len(down_children) == 1:
                                # If the considered child have been completely filled with water from the parent
                                if exported_water[p] - self.xylem_water[down_children[0]] > 0:
                                    # The exposition time is longer if the water content of the target neighbour is more important.
                                    self.cumulated_radial_exchanges_Nm[down_children[0]] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * self.xylem_water[down_children[0]] / self.xylem_water[v]
                                    self.cumulated_radial_exchanges_AA[down_children[0]] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * self.xylem_water[down_children[0]] / self.xylem_water[v]
                                    children_exported_water += [exported_water[p] - self.xylem_water[down_children[0]]]
                                else:
                                    self.cumulated_radial_exchanges_Nm[down_children[0]] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * exported_water[p] / self.xylem_water[v]
                                    self.cumulated_radial_exchanges_AA[down_children[0]] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * exported_water[p] / self.xylem_water[v]
                                    # If all water content from initial segment gone through this axis is exported to the considered child
                                    if exported_water[p] > self.xylem_water[v] * axis_proportion[p]:
                                        self.displaced_Nm_in[down_children[0]] += self.displaced_Nm_out[v] * axis_proportion[p]
                                        self.displaced_AA_in[down_children[0]] += self.displaced_AA_out[v] * axis_proportion[p]
                                    else:
                                        # Displaced matter is shared between child and its parent
                                        child_proportion = exported_water[p] / (self.xylem_water[v] * axis_proportion[p])
                                        self.displaced_Nm_in[down_children[0]] += self.displaced_Nm_out[v] * axis_proportion[p] * child_proportion
                                        self.displaced_AA_in[down_children[0]] += self.displaced_AA_out[v] * axis_proportion[p] * child_proportion
                                        self.displaced_Nm_in[parent[p]] += self.displaced_Nm_out[v] * axis_proportion[p] * (1 - child_proportion)
                                        self.displaced_AA_in[parent[p]] += self.displaced_AA_out[v] * axis_proportion[p] * (1 - child_proportion)
                                    # Break the loop
                                    children_exported_water += [0]

                            # Else if there are several children
                            else:
                                # Water repartition is done according to radius,
                                # as this is the main criteria used in the water model
                                radius_sum = sum([self.radius[k] for k in down_children])
                                children_down_flow = [exported_water[p] * self.radius[k] / radius_sum for k in down_children]
                                children_radius_prop = [self.radius[k] / radius_sum for k in down_children]
                                # Actualize the repartition of water when there is a new branching
                                for k in range(1, len(children_radius_prop)):
                                    axis_proportion.insert(p + 1, axis_proportion[p] * children_radius_prop[-k])

                                axis_proportion[p] = axis_proportion[p] * children_radius_prop[0]
                                for ch in range(len(down_children)):
                                    # If the considered child have been completely filled with water from the parent
                                    if children_down_flow[ch] - self.xylem_water[down_children[ch]] > 0 :
                                        # The exposition time is longer if the water content of the target neighbour is more important.
                                        self.cumulated_radial_exchanges_Nm[down_children[ch]] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * self.xylem_water[down_children[ch]] / self.xylem_water[v]
                                        self.cumulated_radial_exchanges_AA[down_children[ch]] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * self.xylem_water[down_children[ch]] / self.xylem_water[v]
                                        children_down_flow[ch] -= self.xylem_water[down_children[ch]]
                                    else:
                                        self.cumulated_radial_exchanges_Nm[down_children[ch]] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * children_down_flow[ch] / self.xylem_water[v]
                                        self.cumulated_radial_exchanges_AA[down_children[ch]] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * children_down_flow[ch] / self.xylem_water[v]
                                        # If all water content from initial segment gone through this axis is exported to the considered child
                                        if children_down_flow[ch] > self.xylem_water[v] * axis_proportion[p + ch]:
                                            self.displaced_Nm_in[down_children[ch]] += self.displaced_Nm_out[v] * axis_proportion[p + ch]
                                            self.displaced_AA_in[down_children[ch]] += self.displaced_AA_out[v] * axis_proportion[p + ch]
                                        else:
                                            # Displaced matter is shared between child and its parent
                                            child_proportion = children_down_flow[ch] / (self.xylem_water[v] * axis_proportion[p + ch])
                                            self.displaced_Nm_in[down_children[ch]] += self.displaced_Nm_out[v] * axis_proportion[p + ch] * child_proportion
                                            self.displaced_AA_in[down_children[ch]] += self.displaced_AA_out[v] * axis_proportion[p + ch] * child_proportion
                                            self.displaced_Nm_in[parent[p]] += self.displaced_Nm_out[v] * axis_proportion[p + ch] * (1 - child_proportion)
                                            self.displaced_AA_in[parent[p]] += self.displaced_AA_out[v] * axis_proportion[p + ch] * (1 - child_proportion)
                                        # Break the loop
                                        children_down_flow[ch] = 0
                                children_exported_water += children_down_flow

                            # Concatenate so that each children will become parent for the next loop
                            children_list += down_children

                    # Children become parent for the next loop
                    parent = children_list
                    exported_water = children_exported_water

        # If this is an inflow from both up an down segments
        if self.axial_import_water_down[v] >= 0 >= self.axial_export_water_up[v]:
            # There is no exported matter, thus no receiver
            self.displaced_Nm_out[v] = 0
            self.displaced_AA_out[v] = 0
            # No matter what the transported water amount is, all radial transport effects remain on the current vertex.
            self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * self.sub_time_step
            self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * self.sub_time_step

    # METABOLIC PROCESSES
    def process_AA_synthesis(self, C_hexose_root, struct_mass, Nm, **kwargs):
        # amino acid synthesis
        if C_hexose_root > 0 and Nm > 0:
            return struct_mass * kwargs["smax_AA"] / (
                    ((1 + kwargs["Km_Nm_AA"]) / Nm) + ((1 + kwargs["Km_C_AA"]) / C_hexose_root))
        else:
            return 0

    # def process_struct_synthesis(self, v, smax_struct, Km_AA_struct, **kwargs):
    #     # Organic structure synthesis (REPLACED BY RHIZODEP struct_mass_produced)
    #     struct_synthesis = struct_mass * (smax_struct * AA / (Km_AA_struct + AA))

    def process_storage_synthesis(self, struct_mass, AA, **kwargs):
        # Organic storage synthesis (Michaelis-Menten kinetic)
        return struct_mass * (kwargs["smax_stor"] * AA / (kwargs["Km_AA_stor"] + AA))

    def process_storage_catabolism(self, struct_mass, C_hexose_root, C_hexose_reserve, **kwargs):
        # Organic storage catabolism through proteinase
        Km_stor_root = kwargs["Km_stor_catab"] * np.exp(kwargs["storage_C_regulation"] * C_hexose_root)
        return struct_mass * kwargs["cmax_stor"] * C_hexose_reserve / (Km_stor_root + C_hexose_reserve)

    def process_AA_catabolism(self, C_hexose_root, struct_mass, AA, **kwargs):
        # AA catabolism through GDH
        Km_stor_root = kwargs["Km_AA_catab"] * np.exp(kwargs["storage_C_regulation"] * C_hexose_root)
        return struct_mass * kwargs["cmax_AA"] * AA / (Km_stor_root + AA)

    def process_cytokinin_synthesis(self, total_struct_mass, total_hexose, total_Nm, **kwargs):
        return total_struct_mass * kwargs["smax_cytok"] * (
                total_hexose / (total_hexose + kwargs["Km_C_cytok"])) * (
                total_Nm / (total_Nm + kwargs["Km_N_cytok"]))

    # UPDATE NITROGEN POOLS
    def update_Nm(self, Nm, struct_mass, import_Nm, diffusion_Nm_soil, diffusion_Nm_xylem, export_Nm, AA_synthesis, AA_catabolism, **kwargs):
        if struct_mass > 0:
            return Nm + (self.sub_time_step / struct_mass) * (
                    import_Nm
                    - diffusion_Nm_soil
                    + diffusion_Nm_xylem
                    - export_Nm
                    - AA_synthesis * kwargs["r_Nm_AA"]
                    + AA_catabolism / kwargs["r_Nm_AA"])
        else:
            return 0

    def update_AA(self, AA, struct_mass, diffusion_AA_phloem, import_AA, diffusion_AA_soil, export_AA, AA_synthesis,
                  struct_synthesis, storage_synthesis, storage_catabolism, AA_catabolism, struct_mass_produced, **kwargs):
        if struct_mass > 0:
            return AA + (self.sub_time_step / struct_mass) * (
                    diffusion_AA_phloem
                    + import_AA
                    - diffusion_AA_soil
                    - export_AA
                    + AA_synthesis
                    - struct_synthesis * kwargs["r_AA_struct"]
                    - storage_synthesis * kwargs["r_AA_stor"]
                    + storage_catabolism / kwargs["r_AA_stor"]
                    - AA_catabolism
            ) - struct_mass_produced * 0.2 / 146
        # glutamine 5 C -> 60g.mol-1 2N -> 28 g.mol-1 : C:N = 2.1
        # Sachant C:N struct environ de 10 = (Chex + CAA)/NAA Chex = 10*28 - 60 = 220 g Chex.
        # Sachang qu'un hexose contient 12*6=72 gC.mol-1 hex, c'est donc environ 3 hexoses pour 1 AA qui seraient consommés.
        # La proportion d'AA consommée par g de struct mass est donc de 1*146/(3*180 + 1*146) = 0.2 (180 g.mol-1 pour le glucose)

        else:
            return 0

    # def update_struct_protein(self, v, **kwargs):
    #     struct_protein += (sub_time_step / struct_mass) * (
    #         struct_synthesis)

    def update_storage_protein(self, storage_protein, struct_mass, storage_synthesis, storage_catabolism, **kwargs):
        if struct_mass > 0:
            return storage_protein + (self.sub_time_step / struct_mass) * (
                    storage_synthesis
                    - storage_catabolism
            )
        else:
            return 0

    def update_xylem_Nm(self, xylem_Nm, displaced_Nm_in, displaced_Nm_out, cumulated_radial_exchanges_Nm, struct_mass, **kwargs):
        if struct_mass > 0:
            # Vessel's nitrogen pool update
            # Xylem balance accounting for exports from all neighbors accessible by water flow
            return xylem_Nm + (displaced_Nm_in - displaced_Nm_out + cumulated_radial_exchanges_Nm) / struct_mass
        else:
            return 0

    def update_xylem_AA(self, xylem_AA, displaced_AA_in, displaced_AA_out, cumulated_radial_exchanges_AA, struct_mass, **kwargs):
        if struct_mass > 0:
            return xylem_AA + (displaced_AA_in - displaced_AA_out + cumulated_radial_exchanges_AA) / struct_mass
        else:
            return 0

    # PLANT SCALE PROPERTIES UPDATE
    def actualize_total_phloem_AA(self, total_phloem_AA, diffusion_AA_phloem, AA_root_shoot_phloem, total_struct_mass, **kwargs):
        return total_phloem_AA[1] + (- self.sub_time_step * sum(diffusion_AA_phloem.values()) + AA_root_shoot_phloem[1]) / (
                total_struct_mass[1] * kwargs["phloem_cross_area_ratio"])

    def actualize_total_cytokinins(self, total_cytokinins, cytokinin_synthesis, cytokinins_root_shoot_xylem,
                                   total_struct_mass, **kwargs):
        return total_cytokinins[1] + (cytokinin_synthesis[1] * self.sub_time_step -
                                     cytokinins_root_shoot_xylem[1]) / total_struct_mass[1]

    # UPDATE CUMULATIVE VALUES
    # TODO : Retrieve the total struct mass from Rhizodep, otherwise, computation order is messed up here.
    def actualize_total_struct_mass(self, struct_mass, **kwargs):
        # WARNING, do not parallelize otherwise other pool updates will be based on previous time-step
        return sum(struct_mass.values())

    def actualize_total_Nm(self, Nm, struct_mass, total_struct_mass, **kwargs):
        return sum([x*y for x, y in zip(Nm.values(), struct_mass.values())]) / total_struct_mass[1]

    def actualize_total_AA(self, AA, struct_mass, total_struct_mass, **kwargs):
        return sum([x * y for x, y in zip(AA.values(), struct_mass.values())]) / total_struct_mass[1]

    def actualize_total_xylem_Nm(self, xylem_Nm, xylem_struct_mass, total_struct_mass, **kwargs):
        return sum([x*y for x, y in zip(xylem_Nm.values(), xylem_struct_mass.values())]) / total_struct_mass[1]

    def actualize_total_xylem_AA(self, xylem_AA, xylem_struct_mass, total_struct_mass, **kwargs):
        return sum([x*y for x, y in zip(xylem_AA.values(), xylem_struct_mass.values())]) / total_struct_mass[1]

    def actualize_total_AA_rhizodeposition(self, diffusion_AA_soil, import_AA, **kwargs):
        return self.sub_time_step * (sum(diffusion_AA_soil.values()) - sum(import_AA.values()))

    def actualize_total_hexose(self, C_hexose_root, struct_mass, total_struct_mass, **kwargs):
        return sum([x*y for x, y in zip(C_hexose_root.values(), struct_mass.values())]) / total_struct_mass[1]

    # UPDATE STRUCTURAL VALUES TODO : do not keep in this module
    def update_xylem_struct_mass(self, struct_mass, **kwargs):
        return struct_mass * kwargs["xylem_cross_area_ratio"]

    def update_phloem_struct_mass(self, struct_mass, **kwargs):
        return struct_mass * kwargs["phloem_cross_area_ratio"]

    # CARBON UPDATE
    def update_C_hexose_root(self, C_hexose_root, **kwargs):
        # Minimum to avoid issues with zero values
        if C_hexose_root <= 0:
            return 1e-1
        else:
            return C_hexose_root

