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

# Dataclass for initialisation and parametrization.
# For readability's sake, only units are displayed. See functions' documentation for descriptions.


# Properties' initialization


@dataclass
class InitCommonN:
    # time resolution
    sub_time_step: int = 3600  # (second) MUST be a multiple of base time_step
    # Pools initial size
    Nm: float = 1e-4    # mol N.g-1
    AA: float = 9e-4    # mol AA.g-1
    struct_protein: float = 0   # mol prot struct.g-1
    storage_protein: float = 0     # mol prot stor.g-1
    xylem_Nm: float = 1e-4  # mol N.g-1
    xylem_AA: float = 1e-4  # mol AA.g-1
    # Transport processes
    import_Nm: float = 0    # mol N.s-1
    import_AA: float = 0    # mol AA.s-1
    export_Nm: float = 0    # mol N.s-1
    export_AA: float = 0    # mol AA.s-1
    diffusion_Nm_soil: float = 0    # mol N.s-1
    diffusion_Nm_xylem: float = 0    # mol N.s-1
    diffusion_Nm_soil_xylem: float = 0  # mol N.s-1
    diffusion_AA_soil: float = 0  # mol AA.s-1
    diffusion_AA_phloem: float = 0    # mol AA.s-1
    diffusion_AA_soil_xylem: float = 0  # mol AA.s-1
    # Metabolic processes
    AA_synthesis: float = 0    # mol AA.s-1
    struct_synthesis: float = 0    # mol struct.s-1
    storage_synthesis: float = 0    # mol stor.s-1
    AA_catabolism: float = 0    # mol AA.s-1
    storage_catabolism: float = 0    # mol stor.s-1
    phloem_total_AA: float = 9e-4  # mol AA.g-1
    total_cytokinins: float = 100  # Artif UA.g-1
    cytokinin_synthesis: float = 0   # UA cytokinin.s-1


@dataclass
class InitDiscreteVesselsN(InitCommonN):
    xylem_struct_mass: float = 1e-6  # g
    displaced_Nm_in: float = 0   # mol Nm.time_step-1
    displaced_Nm_out: float = 0  # mol Nm.time_step-1
    displaced_AA_in: float = 0  # mol Nm.time_step-1
    displaced_AA_out: float = 0  # mol Nm.time_step-1
    cumulated_radial_exchanges_Nm: float = 0  # mol Nm.time_step-1
    cumulated_radial_exchanges_AA: float = 0  # mol AA.time_step-1
    phloem_struct_mass: float = 5e-7  # g

# Parameters' default value


@dataclass
class TransportCommonN:
    # kinetic parameters
    vmax_Nm_root: float = 1e-7     # mol N.s-1.m-2
    vmax_Nm_xylem: float = 1e-7     # mol N.s-1.m-2
    Km_Nm_root_LATS: float = 1e-1   # mol N.m-3 Changed to increase diminution
    Km_Nm_root_HATS: float = 1e-6  # mol N.m-3
    begin_N_regulation: float = 1e1   # Artif mol N.g-1 changed so that import_Nm variation may occur in Nm variation range
    span_N_regulation: float = 2e-4    # mol N.g-1 range corresponding to observed variation range within segment
    Km_Nm_xylem: float = 8e-5   # mol N.g-1
    vmax_AA_root: float = 1e-7     # mol AA.s-1.m-2
    Km_AA_root: float = 1e-3    # mol AA.m-3
    vmax_AA_xylem: float = 1e-7     # mol AA.s-1.m-2
    Km_AA_xylem: float = 1e-3   # mol AA.g-1
    diffusion_soil: float = 1e-12   # Artif g.m-2.s-1 while there is no soil model balance
    diffusion_xylem: float = 1e-4   # g.m-2.s-1 more realistic ranges
    diffusion_phloem: float = 1e-5  # Artif -1 g.m-2.s-1 more realistic ranges
    diffusion_apoplasm: float = 2.5e-10  # Artif. g.m-2.s-1 while there is no soil model balance
    # metabolism-related parameters
    transport_C_regulation: float = 7e-3    # mol.g-1


@dataclass
class TransportAxialN(TransportCommonN):
    None


@dataclass
class MetabolismN:
    # TODO : introduce nitrogen fixation
    # kinetic parameters
    smax_AA: float = 1e-6   # Artif mol.s-1.g-1 DW
    Km_Nm_AA: float = 3e-6  # mol.g-1 DW
    Km_C_AA: float = 350e-6     # mol.g-1 DW
    smax_struct: float = 1e-9    # mol.s-1.g-1 DW
    Km_AA_struct: float = 250e-6    # mol.g-1 DW
    smax_stor: float = 0  # 1e-9  # mol.s-1.g-1 DW 0 for wheat
    Km_AA_stor: float = 250e-6    # mol.g-1 DW
    cmax_stor: float = 1e-9   # mol.s-1.g-1 DW
    Km_stor_catab: float = 250e-6    # mol.g-1 DW
    cmax_AA: float = 1.2e-8    # mol.s-1.g-1 DW
    Km_AA_catab: float = 2.5e-6     # mol.g-1 DW
    storage_C_regulation: float = 7e-3  # mol.g-1

@dataclass
class MetabolismHormones:
    # kinetic parameters
    smax_cytok: float = 9e-4  # UA.g DW-1.s-1
    Km_C_cytok: float = 1.2e-3
    Km_N_cytok:float = 5e-5


@dataclass
class UpdateN:
    r_Nm_AA: float = 1.4
    r_AA_struct: float = 65
    r_AA_stor: float = 65
    xylem_cross_area_ratio: float = 0.84 * (0.36 ** 2)  # (adim) apoplasmic cross-section area ratio * stele radius ratio^2
    phloem_cross_area_ratio: float = 0.15 * (0.36 ** 2)  # (adim) phloem cross-section area ratio * stele radius ratio^2


# Nitrogen Model versions as classes. A version relates to a set of structural assumptions given in the class name.
# OnePoolVessels class has been discontinued


class CommonNitrogenModel:
    def __init__(self, g, time_step, sub_time_step, Nm, AA, struct_protein, storage_protein, import_Nm, import_AA, export_Nm, export_AA, diffusion_Nm_soil,
                 diffusion_Nm_xylem, diffusion_Nm_soil_xylem, diffusion_AA_soil, diffusion_AA_phloem, diffusion_AA_soil_xylem, AA_synthesis, struct_synthesis,
                 storage_synthesis, AA_catabolism, storage_catabolism, phloem_total_AA, total_cytokinins, cytokinin_synthesis):

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
        self.time_step = time_step
        self.sub_time_step = sub_time_step

        # New properties' creation in MTG
        self.keywords.update(dict(Nm=Nm,
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
                        storage_catabolism=storage_catabolism
                        ))

        props = self.g.properties()
        for name in self.keywords:
            props.setdefault(name, {})

        # vertices storage for future calls in for loops
        self.vertices = self.g.vertices(scale=g.max_scale())
        for vid in self.vertices:
            for name, value in self.keywords.items():
                # Effectively creates the new property
                props[name][vid] = value

        # Accessing properties once, pointing to g for further modifications
        self.states += """
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
                        length
                        radius
                        struct_mass
                        C_hexose_root
                        C_hexose_reserve
                        C_hexose_reserve
                        living_root_hairs_external_surface
                        thermal_time_since_emergence
                        """.split()

        # Declare MTG properties in self
        for name in self.states:
            setattr(self, name, props[name])

        # Repeat the same process for total root system properties

        # Creating variables for
        self.totals_keywords.update(dict(total_Nm=0,
                                    total_AA=0,
                                    total_hexose=0,
                                    total_cytokinins=total_cytokinins,
                                    total_struct_mass=0,
                                    xylem_total_Nm=0,
                                    xylem_total_AA=0,
                                    phloem_total_AA=phloem_total_AA,
                                    Nm_root_shoot_xylem=0,
                                    AA_root_shoot_xylem=0,
                                    total_AA_rhizodeposition=0,
                                    cytokinin_synthesis=cytokinin_synthesis
                                    ))

        for name, value in self.totals_keywords.items():
            props.setdefault(name, {})
            props[name][1] = value

        # Accessing properties once, pointing to g for further modifications
        self.totals_states += """
                                    total_Nm
                                    total_AA
                                    total_hexose
                                    total_cytokinins
                                    total_struct_mass
                                    xylem_total_Nm
                                    xylem_total_AA
                                    phloem_total_AA
                                    Nm_root_shoot_xylem
                                    AA_root_shoot_xylem
                                    total_AA_rhizodeposition
                                    cytokinin_synthesis
                                    """.split()

        # Declare MTG properties in self
        for name in self.totals_states:
            setattr(self, name, props[name])

        # Declare to outside modules which variables are needed
        self.inputs.update({
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
            ]
        })

    def transport_radial_N(self, v, vmax_Nm_root, vmax_Nm_xylem, Km_Nm_root_LATS, Km_Nm_root_HATS, begin_N_regulation, span_N_regulation,
                        Km_Nm_xylem, vmax_AA_root, Km_AA_root, vmax_AA_xylem, Km_AA_xylem, diffusion_soil, diffusion_xylem, diffusion_phloem, diffusion_apoplasm,
                        transport_C_regulation):

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

        # ONLY RADIAL TRANSPORT IS COMMON BETWEEN MODELS

        # MINERAL NITROGEN TRANSPORT

        # We define mineral nitrogen active uptake from soil

        precision = 0.99
        Km_Nm_root = (Km_Nm_root_LATS - Km_Nm_root_HATS)/(
                        1 + (precision/((1-precision) * np.exp(-begin_N_regulation))
                        * np.exp(-self.Nm[v]/span_N_regulation))
                        ) + Km_Nm_root_HATS

        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        self.import_Nm[v] = ((self.soil_Nm[v] * vmax_Nm_root / (self.soil_Nm[v] + Km_Nm_root))
                            * (self.root_exchange_surface[v] + self.living_root_hairs_external_surface[v])
                            * (self.C_hexose_root[v] / (self.C_hexose_root[v] + transport_C_regulation)))

        # Passive radial diffusion between soil and cortex.
        # It happens only through root segment external surface.
        # We summarize apoplasm-soil and cortex-soil diffusion in 1 flow.
        self.diffusion_Nm_soil[v] = (diffusion_soil * (self.Nm[v]*10e5 - self.soil_Nm[v])
                            * (self.root_exchange_surface[v] + self.living_root_hairs_external_surface[v]))

        # We define active export to xylem from root segment

        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        self.export_Nm[v] = ((self.Nm[v] * vmax_Nm_xylem / (self.Nm[v] + Km_Nm_xylem))
                                * self.stele_exchange_surface[v]
                                * (self.C_hexose_root[v] / (self.C_hexose_root[v] + transport_C_regulation)))

        # Passive radial diffusion between xylem and cortex through plasmalema
        self.diffusion_Nm_xylem[v] = (diffusion_xylem * (self.xylem_Nm[v] - self.Nm[v])
                                    * self.stele_exchange_surface[v])
        
        # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
        # Here, surface is not really representative of a structure as everything is apoplasmic
        self.diffusion_Nm_soil_xylem[v] = (diffusion_apoplasm * (self.soil_Nm[v] - self.xylem_Nm[v]*10e5)
                                    * 2 * np.pi * self.radius[v] * self.length[v] * self.apoplasmic_stele[v])

        # AMINO ACID TRANSPORT

        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        self.import_AA[v] = ((self.soil_AA[v] * vmax_AA_root / (self.soil_AA[v] + Km_AA_root))
                             * (self.root_exchange_surface[v] + self.living_root_hairs_external_surface[v])
                             * (self.C_hexose_root[v] / (self.C_hexose_root[v] + transport_C_regulation)))

        # We define amino acid passive diffusion to soil
        self.diffusion_AA_soil[v] = (diffusion_soil * (self.AA[v]*10e5 - self.soil_AA[v])
                                    * (self.root_exchange_surface[v] + self.living_root_hairs_external_surface[v]))

        # We define active export to xylem from root segment

        # Km is defined as a constant here
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        self.export_AA[v] = ((self.AA[v] * vmax_AA_xylem / (self.AA[v] + Km_AA_xylem))
                                * self.stele_exchange_surface[v]
                                * (self.C_hexose_root[v] / (
                        self.C_hexose_root[v] + transport_C_regulation)))

        # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
        self.diffusion_AA_soil_xylem[v] = (diffusion_apoplasm * (self.soil_AA[v] - self.xylem_AA[v]*10e5)
                                    * 2 * np.pi * self.radius[v] * self.length[v] * self.apoplasmic_stele[v])

        # Passive radial diffusion between phloem and cortex through plasmodesmata
        self.diffusion_AA_phloem[v] = (diffusion_phloem * (self.phloem_total_AA[1] - self.AA[v])
                                       * self.phloem_exchange_surface[v])

    def metabolism_N(self, v, smax_AA, Km_Nm_AA, Km_C_AA, smax_struct, Km_AA_struct, smax_stor,
                     Km_AA_stor, cmax_stor, Km_stor_catab, cmax_AA, Km_AA_catab, storage_C_regulation):

        """
        Description
        ___________
        Nitrogen metabolism within local root segment cortex

        Parameters
        __________
        :param smax_struct : Maximal organic structure synthesis from mineral nitrogen and labil C (mol.s-1)
        :param Km_Nm_struct :
        :param Km_C_struct :

        """

        # amino acid synthesis
        if self.C_hexose_root[v] > 0:
            self.AA_synthesis[v] = self.struct_mass[v] * smax_AA / (
                ((1 + Km_Nm_AA) / self.Nm[v])
                + ((1 + Km_C_AA) / self.C_hexose_root[v])
            )
        else:
            self.AA_synthesis[v] = 0.0

        # Organic structure synthesis
        self.struct_synthesis[v] = self.struct_mass[v] * (smax_struct * self.AA[v]
                                       / (Km_AA_struct + self.AA[v]))

        # Organic storage synthesis (Michaelis-Menten kinetic)
        self.storage_synthesis[v] = self.struct_mass[v] * (smax_stor * self.AA[v]
                                       / (Km_AA_stor + self.AA[v]))

        # Organic storage catabolism through proteinase
        Km_stor_root = Km_stor_catab * np.exp(storage_C_regulation * self.C_hexose_root[v])
        self.storage_catabolism[v] = self.struct_mass[v] * cmax_stor * self.C_hexose_reserve[v] / (
                Km_stor_root + self.C_hexose_reserve[v])

        # AA catabolism through GDH
        Km_stor_root = Km_AA_catab * np.exp(storage_C_regulation * self.C_hexose_root[v])
        self.AA_catabolism[v] = self.struct_mass[v] * cmax_AA * self.AA[v] / (
                Km_stor_root + self.AA[v])

    def transport_C(self, v, apex_C_hexose_root=0.4, hexose_decrease_rate=0.3):
        # artificially fixated hexose concentration before coupling with C model
        self.C_hexose_root[v] = apex_C_hexose_root - hexose_decrease_rate * self.thermal_time_since_emergence[v] / max(
            self.thermal_time_since_emergence.values())

    def metabolism_total_hormones(self, smax_cytok, Km_C_cytok, Km_N_cytok):
        self.cytokinin_synthesis[0] = self.total_struct_mass[1] * smax_cytok * (
                self.total_hexose[1]/(self.total_hexose[1] + Km_C_cytok)) * (
                self.total_Nm[1]/(self.total_Nm[1] + Km_N_cytok))

    def update_N_local(self, v, r_Nm_AA, r_AA_struct, r_AA_stor, phloem_cross_area_ratio):

        self.Nm[v] += (self.sub_time_step / self.struct_mass[v]) * (
                self.import_Nm[v]
                - self.diffusion_Nm_soil[v]
                + self.diffusion_Nm_xylem[v]
                - self.export_Nm[v]
                - self.AA_synthesis[v] * r_Nm_AA
                + self.AA_catabolism[v] / r_Nm_AA)

        self.AA[v] += (self.sub_time_step / self.struct_mass[v]) * (
                self.diffusion_AA_phloem[v]
                + self.import_AA[v]
                - self.diffusion_AA_soil[v]
                - self.export_AA[v]
                + self.AA_synthesis[v]
                - self.struct_synthesis[v] * r_AA_struct
                - self.storage_synthesis[v] * r_AA_stor
                + self.storage_catabolism[v] / r_AA_stor
                - self.AA_catabolism[v]
                )

        if self.AA[v] < 0:
            print("error")

        self.struct_protein[v] += (self.sub_time_step / self.struct_mass[v]) * (
                self.struct_synthesis[v]
                )

        self.storage_protein[v] += (self.sub_time_step / self.struct_mass[v]) * (
                self.storage_synthesis[v]
                - self.storage_catabolism[v]
                )

        self.phloem_total_AA[1] += - (self.sub_time_step * self.diffusion_AA_phloem[v]) / (self.total_struct_mass[1] * phloem_cross_area_ratio)

    def update_sums(self):
        self.total_struct_mass[1] = sum(self.struct_mass.values())
        self.total_Nm[1] = sum([x*y for x, y in zip(self.Nm.values(), self.struct_mass.values())]) / self.total_struct_mass[1]
        self.total_AA[1] = sum([x * y for x, y in zip(self.AA.values(), self.struct_mass.values())]) / self.total_struct_mass[1]
        self.total_AA_rhizodeposition[1] = self.sub_time_step * (sum(self.diffusion_AA_soil.values()) - sum(self.import_AA.values()))
        self.total_hexose[1] = sum([x*y for x, y in zip(self.C_hexose_root.values(), self.struct_mass.values())]) / self.total_struct_mass[1]
        self.total_cytokinins[1] += (self.cytokinin_synthesis[1] * self.sub_time_step - self.cytokinins_root_shoot_xylem[1]) / self.total_struct_mass[1]


class DiscreteVessels(CommonNitrogenModel):

    def __init__(self, g, time_step, xylem_Nm, xylem_AA, xylem_struct_mass, displaced_Nm_in, displaced_Nm_out, displaced_AA_in,
                 displaced_AA_out, cumulated_radial_exchanges_Nm, cumulated_radial_exchanges_AA, phloem_struct_mass, **kwargs):

        self.g = g

        # New properties' creation in MTG
        self.keywords = dict(xylem_Nm=xylem_Nm,
                            xylem_AA=xylem_AA,
                            xylem_struct_mass=xylem_struct_mass,
                            displaced_Nm_in=displaced_Nm_in,
                            displaced_Nm_out=displaced_Nm_out,
                            displaced_AA_in=displaced_AA_in,
                            displaced_AA_out=displaced_AA_out,
                            cumulated_radial_exchanges_Nm=cumulated_radial_exchanges_Nm,
                            cumulated_radial_exchanges_AA=cumulated_radial_exchanges_AA,
                            phloem_struct_mass=phloem_struct_mass)

        self.totals_keywords = {}

        # Properties to be accessed, pointing to g for further modifications
        self.states = """
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
                """.split()

        self.totals_states = []

        self.inputs = {
            "water": [
                "xylem_water",
                "axial_export_water_up",
                "axial_import_water_down"
            ]
        }

        super().__init__(g, time_step, **kwargs)

    def transport_N(self, v, **kwargs):
        """
                Description
                ___________

        """
        # TODO : probably collar children have to be reintroduced for good neighbor management. (from model_water)
        #  But if null water content proprely passes information between collar and it's children,
        #  it may be already working well

        # RADIAL TRANSPORT

        self.transport_radial_N(v=v, **kwargs)

        # AXIAL TRANSPORT

        # If this is only an out flow to up parents
        if self.axial_export_water_up[v] > 0:
            # Turnover defines a dilution factor of radial transport processes over the axially transported
            # water column
            turnover = self.axial_export_water_up[v] / self.xylem_water[v]
            if turnover <= 1:
                #print("Uturnover <1")
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
                #print("D<1")
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

                            # Else if there are several children.
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
                                print(axis_proportion, sum(axis_proportion))
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

    def update_N(self, r_Nm_AA, r_AA_struct, r_AA_stor, xylem_cross_area_ratio, phloem_cross_area_ratio):
        """
        Description
        ___________
        This function aims at computing the local and global root nitrogen balance.

        For xylem, the computational loop is the following :
        TODO
        - Starting from the previous xylem mean concentrations and time-step water flows, repartition of root segment
        radial exchanges within xylem is operated.
        - Then the resulting export to shoot is computed from the previous mean concentrations.
        - Finally, local balance and export to shoot are used to compute the xylem mean concentrations used for
        the next time-step.

        Parameters
        __________
        See root_cynaps.model_nitrogen.CommonNitrogenModel.update_N_local() for descriptions about passed parameters.
        """

        # Update plant-level properties first as we need to compute the total mass
        self.update_sums()

        # for all root segments in MTG...
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:

                # Local nitrogen concentration update
                self.update_N_local(vid, r_Nm_AA, r_AA_struct, r_AA_stor, phloem_cross_area_ratio)

                # Local vessels' structural mass update
                self.xylem_struct_mass[vid] = self.struct_mass[vid] * xylem_cross_area_ratio
                self.phloem_struct_mass[vid] = self.struct_mass[vid] * phloem_cross_area_ratio

                # Vessel's nitrogen pool update
                # Xylem balance accounting for exports from all neighbors accessible by water flow
                # Hypothesis : by using the mean concentration we suppose a constitutive homogeneisation of xylem concentrations if no flow or loading occurs
                self.xylem_Nm[vid] += (self.displaced_Nm_in[vid] - self.displaced_Nm_out[vid] + self.cumulated_radial_exchanges_Nm[vid]) / self.struct_mass[vid]

                self.xylem_AA[vid] += (self.displaced_AA_in[vid] - self.displaced_AA_out[vid] + self.cumulated_radial_exchanges_AA[vid]) / self.struct_mass[vid]



        # Update vessel scale properties
        self.xylem_total_Nm[1] = sum([x*y for x, y in zip(self.xylem_Nm.values(), self.xylem_struct_mass.values())]) / self.total_struct_mass[1]
        self.xylem_total_AA[1] = sum([x*y for x, y in zip(self.xylem_AA.values(), self.xylem_struct_mass.values())]) / self.total_struct_mass[1]
        self.phloem_total_AA[1] += self.AA_root_shoot_phloem[1] / (self.total_struct_mass[1] * phloem_cross_area_ratio)

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

    def exchanges_and_balance(self, hexose_decrease_rate):

        """
        Description
        ___________
        Model time-step processes and balance for nitrogen to be called by simulation files.

        """

        # Computing all derivative processes
        # Global root system processes
        self.metabolism_total_hormones(**asdict(MetabolismHormones()))
        self.initialize_cumulative()
        # For each sub_time_step
        for k in range(int(self.time_step/self.sub_time_step)):

            # Spatialized for all root segments in MTG...
            for vid in self.vertices:
                # if root segment emerged
                if self.struct_mass[vid] > 0:
                    self.transport_C(vid, hexose_decrease_rate=hexose_decrease_rate)
                    self.transport_N(vid, **asdict(TransportAxialN()))
                    self.metabolism_N(vid, **asdict(MetabolismN()))

            self.update_N(**asdict(UpdateN()))
