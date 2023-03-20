"""
rhizodep.nitrogen
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

OnePoolVessels(CommonNitrogenModel)

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
    # Pools initial size
    Nm: float = 1e-4    # mol N.g-1
    AA: float = 1e-4    # mol AA.g-1
    struct_protein: float = 0   # mol prot struct.g-1
    storage_protein: float = 0     # mol prot stor.g-1
    xylem_Nm: float = 1e-4  # mol N.g-1
    xylem_AA: float = 1e-4  # mol AA.g-1
    phloem_AA: float = 1e-4  # mol AA.g-1
    # Transport processes
    import_Nm: float = 0    # mol N.s-1
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


@dataclass
class InitDiscreteVesselsN(InitCommonN):
    xylem_struct_mass: float = 1e-3  # g
    phloem_struct_mass: float = 1e-3  # g
    axial_diffusion_Nm_xylem: float = 0    # mol N.s-1
    axial_diffusion_AA_xylem: float = 0    # mol AA.s-1
    axial_diffusion_AA_phloem: float = 0    # mol AA.s-1


# Parameters' default value


@dataclass
class TransportCommonN:
    # kinetic parameters
    vmax_Nm_root: float = 1e-9     # mol N.s-1.m-2
    vmax_Nm_xylem: float = 1e-9     # mol N.s-1.m-2
    Km_Nm_root_LATS: float = 1e-2    # mol N.g-1
    Km_Nm_root_HATS: float = 1e-4    # mol N.g-1
    begin_N_regulation: float = 1e-3   # mol N.g-1 value
    span_N_regulation: float = 1e-3    # mol N.g-1 range
    Km_Nm_xylem: float = 1e-4   # mol N.g-1
    vmax_AA_xylem: float = 1e-9     # mol AA.s-1.m-2
    Km_AA_xylem: float = 1e-4   # mol AA.g-1
    diffusion_soil: float = 1e-9
    diffusion_xylem: float = 1e-8
    diffusion_phloem: float = 1e-7
    diffusion_apoplasm: float = 1e-7
    # metabolism-related parameters
    transport_C_regulation: float = 1e-2


@dataclass
class TransportAxialN(TransportCommonN):
    # architecture parameters
    xylem_to_root: float = 0.2  # adim
    phloem_to_root: float = 0.15    # adim
    # kinetic parameters
    axial_diffusion_xylem: float = 1e-7
    axial_diffusion_phloem: float = 1e-7



@dataclass
class MetabolismN:
    # kinetic parameters
    smax_AA: float = 0
    Km_Nm_AA: float = 0.001
    Km_C_AA: float = 0.001
    smax_struct: float = 0
    Km_AA_struct: float = 0.001
    smax_stor: float = 0
    Km_AA_stor: float = 0.001
    cmax_stor: float = 0
    Km_stor_catab: float = 0.001
    cmax_AA: float = 0
    Km_AA_catab: float = 0.001
    storage_C_regulation: float = 0.1

@dataclass
class MetabolismHormones:
    # kinetic parameters
    smax_cytok: float = 0.01
    Km_C_cytok: float = 0.001
    Km_N_cytok:float = 0.001


@dataclass
class UpdateN:
    r_Nm_AA: float = 2
    r_AA_struct: float = 2
    r_AA_stor: float = 2
    xylem_to_root: float = 0.2
    phloem_to_root: float = 0.15


# Nitrogen Model versions as classes. A version relates to a set of structural assumptions given in the class name.


class CommonNitrogenModel:
    def __init__(self, g, Nm, AA, struct_protein, storage_protein, import_Nm, export_Nm, export_AA, diffusion_Nm_soil,
                 diffusion_Nm_xylem, diffusion_Nm_soil_xylem, diffusion_AA_soil, diffusion_AA_phloem, diffusion_AA_soil_xylem, AA_synthesis, struct_synthesis,
                 storage_synthesis, AA_catabolism, storage_catabolism, Nm_root_shoot_xylem, AA_root_shoot_xylem,
                 AA_root_shoot_phloem, cytokinins_root_shoot_xylem):

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

        # New properties' creation in MTG
        self.keywords.update(dict(Nm=Nm,
                        AA=AA,
                        struct_protein=struct_protein,
                        storage_protein=storage_protein,
                        import_Nm=import_Nm,
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
                        ))

        # Creating variables for
        self.root_system_totals.update(dict(total_Nm=0,
                                    total_hexose=0,
                                    total_cytokinins=0,
                                    total_struct_mass=0,
                                    xylem_total_Nm=0,
                                    xylem_total_AA=0,
                                    phloem_total_AA=0))

        self.shoot_exchanges = dict(Nm_root_shoot_xylem=Nm_root_shoot_xylem,
                                         AA_root_shoot_xylem=AA_root_shoot_xylem,
                                         AA_root_shoot_phloem=AA_root_shoot_phloem,
                                         cytokinins_root_shoot_xylem=cytokinins_root_shoot_xylem
                                         )

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
                        soil_Nm
                        soil_AA
                        Nm
                        AA
                        struct_protein
                        storage_protein
                        volume
                        import_Nm
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
                        root_exchange_surface
                        stele_exchange_surface
                        apoplasmic_stele
                        length
                        radius
                        struct_mass
                        C_hexose_root
                        C_hexose_reserve
                        thermal_time_since_emergence
                        """.split()

        # Declare MTG properties in self
        for name in self.states:
            setattr(self, name, props[name])

        # Declare exchanges with flow retreived from the shoot model
        for name in self.shoot_exchanges:
            setattr(self, name, self.shoot_exchanges[name])

        # Declare totals computed for global model's outputs
        for name in self.root_system_totals:
            setattr(self, name, self.root_system_totals[name])


    def transport_radial_N(self, v, model, vmax_Nm_root, vmax_Nm_xylem, Km_Nm_root_LATS, Km_Nm_root_HATS, begin_N_regulation, span_N_regulation,
                        Km_Nm_xylem, vmax_AA_xylem, Km_AA_xylem, diffusion_soil, diffusion_xylem, diffusion_phloem, diffusion_apoplasm, 
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
        :param xylem_to_root: Radius ratio between mean xylem and root segment (adim)
        :param phloem_to_root: Radius ratio between mean phloem and root segment (adim)
        :param epiderm_differentiation: Epiderm differentiation rate (°C-1.s-1)
        :param endoderm_differentiation: Endoderm differentiation rate (°C-1.s-1)

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
                        )

        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        self.import_Nm[v] = ((self.soil_Nm[v] * vmax_Nm_root / (self.soil_Nm[v] + Km_Nm_root))
                            * (self.root_exchange_surface[v])
                            * (self.C_hexose_root[v] / (self.C_hexose_root[v] + transport_C_regulation)))

        # Passive radial diffusion between soil and cortex.
        # It happens only through root segment external surface.
        # We summarize apoplasm-soil and cortex-soil diffusion in 1 flow.
        self.diffusion_Nm_soil[v] = (diffusion_soil * (self.Nm[v] - self.soil_Nm[v])
                            * self.root_exchange_surface[v])

        # We define active export to xylem from root segment

        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        self.export_Nm[v] = ((self.Nm[v] * vmax_Nm_xylem / (self.Nm[v] + Km_Nm_xylem))
                                * self.stele_exchange_surface[v]
                                * (self.C_hexose_root[v] / (self.C_hexose_root[v] + transport_C_regulation)))

        # Passive radial diffusion between xylem and cortex through plasmalema
        self.diffusion_Nm_xylem[v] = (diffusion_xylem * (self.xylem_Nm[model] - self.Nm[v])
                                    * self.stele_exchange_surface[v])
        
        # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
        self.diffusion_Nm_soil_xylem[v] = (diffusion_apoplasm * (self.soil_Nm[v] - self.xylem_Nm[model])
                                    * 2 * np.pi * self.radius[v] * self.apoplasmic_stele[v])

        # AMINO ACID TRANSPORT

        # We define amino acid passive diffusion to soil
        self.diffusion_AA_soil[v] = (diffusion_soil * (self.AA[v] - self.soil_AA[v])
                                    * self.root_exchange_surface[v])

        # We define active export to xylem from root segment

        # Km is defined as a constant here because xylem is global
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        self.export_AA[v] = ((self.AA[v] * vmax_AA_xylem / (self.AA[v] + Km_AA_xylem))
                                * self.stele_exchange_surface[v]
                                * (self.C_hexose_root[v] / (
                        self.C_hexose_root[v] + transport_C_regulation)))

        # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
        self.diffusion_AA_soil_xylem[v] = (diffusion_apoplasm * (self.soil_AA[v] - self.xylem_AA[model])
                                    * 2 * np.pi * self.radius[v] * self.apoplasmic_stele[v])

        # Passive radial diffusion between phloem and cortex through plasmodesmata
        self.diffusion_AA_phloem[v] = (diffusion_phloem * (self.phloem_AA[model] - self.AA[v])
                                         * (2 * np.pi * self.radius[v]))

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
            self.AA_synthesis[v] = smax_AA / (
                ((1 + Km_Nm_AA) / self.Nm[v])
                + ((1 + Km_C_AA) / self.C_hexose_root[v])
            )
        else:
            self.AA_synthesis[v] = 0.0

        # Organic structure synthesis
        self.struct_synthesis[v] = (smax_struct * self.AA[v]
                                       / (Km_AA_struct + self.AA[v]))

        # Organic storage synthesis (Michaelis-Menten kinetic)
        self.storage_synthesis[v] = (smax_stor * self.AA[v]
                                       /(Km_AA_stor + self.AA[v]))

        # Organic storage catabolism
        Km_stor_root = Km_stor_catab * np.exp(storage_C_regulation * self.C_hexose_root[v])
        self.storage_catabolism[v] = cmax_stor * self.C_hexose_reserve[v] / (
                Km_stor_root + self.C_hexose_reserve[v])

        # AA catabolism
        Km_stor_root = Km_AA_catab * np.exp(storage_C_regulation * self.C_hexose_root[v])
        self.AA_catabolism[v] = cmax_AA * self.AA[v] / (
                Km_stor_root + self.AA[v])

    def metabolism_total_hormones(self, smax_cytok, Km_C_cytok, Km_N_cytok):
        self.cytokinin_synthesis = smax_cytok * (
                self.total_hexose/(self.total_hexose + Km_C_cytok)) * (
                self.total_Nm/(self.total_Nm + Km_N_cytok))

    def update_N_local(self, v, r_Nm_AA, r_AA_struct, r_AA_stor, time_step):

        self.Nm[v] += time_step / self.struct_mass[v] * (
                self.import_Nm[v]
                - self.diffusion_Nm_soil[v]
                + self.diffusion_Nm_xylem[v]
                - self.export_Nm[v]
                - self.AA_synthesis[v] * r_Nm_AA
                + self.AA_catabolism[v] / r_Nm_AA)

        self.AA[v] += time_step / self.struct_mass[v] * (
                self.diffusion_AA_phloem[v]
                - self.diffusion_AA_soil[v]
                - self.export_AA[v]
                + self.AA_synthesis[v]
                - self.struct_synthesis[v] * r_AA_struct
                - self.storage_synthesis[v] * r_AA_stor
                + self.storage_catabolism[v] / r_AA_stor
                - self.AA_catabolism[v]
                )

        self.struct_protein[v] += time_step/ self.struct_mass[v] * (
                self.struct_synthesis[v]
                )

        self.storage_protein[v] += time_step / self.struct_mass[v] * (
                self.storage_synthesis[v]
                - self.storage_catabolism[v]
                )

    def update_sums(self, time_step):
        self.total_struct_mass = sum(self.struct_mass.values())
        self.total_Nm = sum([x*y for x,y in zip(self.Nm.values(),self.struct_mass.values())])/self.total_struct_mass
        self.total_hexose = sum([x*y for x,y in zip(self.C_hexose_root.values(),self.struct_mass.values())])/self.total_struct_mass
        self.total_cytokinins += time_step / self.total_struct_mass * (
                self.cytokinin_synthesis
                - self.cytokinins_root_shoot_xylem
                )


class OnePoolVessels(CommonNitrogenModel):

    def __init__(self, g, xylem_Nm, xylem_AA, phloem_AA, **kwargs):

        self.keywords = {}
        self.root_system_totals = dict(xylem_Nm=[xylem_Nm],
                                       xylem_AA=[xylem_AA],
                                       phloem_AA=[phloem_AA])
        self.states = []

        super().__init__(g, **kwargs)

    def transport_N(self, v, **kwargs):
        self.transport_radial_N(v, model=0, **kwargs)

    def update_N(self, r_Nm_AA, r_AA_struct, r_AA_stor, xylem_to_root, phloem_to_root, time_step):
        # Get summed root system level properties for outputs
        self.update_sums(time_step)
        # Vessels structural masses, defined as a fixed proportion to be abe to define concentrations
        self.xylem_total_struct_mass = self.total_struct_mass * xylem_to_root
        self.phloem_total_struct_mass = self.total_struct_mass * phloem_to_root
        # for all root segments in MTG...
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:
                # Local nitrogen concentration update
                self.update_N_local(vid, r_Nm_AA, r_AA_struct, r_AA_stor, time_step)

                # Global vessel's nitrogen pool update
                self.xylem_Nm[0] += time_step * (
                            self.export_Nm[vid] + self.diffusion_Nm_soil_xylem[vid] - self.diffusion_Nm_xylem[vid]) / \
                                    self.xylem_total_struct_mass
                self.xylem_AA[0] += time_step * (self.export_AA[vid] + self.diffusion_AA_soil_xylem[vid]) / \
                                    self.xylem_total_struct_mass
                self.phloem_AA[0] -= time_step * self.diffusion_AA_phloem[vid] / self.phloem_total_struct_mass

        # Update vessels according to exchanges with shoot
        self.xylem_Nm[0] -= time_step * self.Nm_root_shoot_xylem / self.xylem_total_struct_mass
        self.xylem_AA[0] -= time_step * self.AA_root_shoot_xylem / self.xylem_total_struct_mass
        self.phloem_AA[0] += time_step * self.AA_root_shoot_phloem / self.phloem_total_struct_mass


        # Global xylem and phloem pools for outputs
        self.xylem_total_Nm = self.xylem_Nm[0]
        self.xylem_total_AA = self.xylem_AA[0]
        self.phloem_total_AA = self.phloem_AA[0]

    def exchanges_and_balance(self, time_step):

        """
        Description
        ___________
        Model time-step processes and balance for nitrogen to be called by simulation files.

        """

        # Computing all derivative processes
        # Global root system processes
        self.metabolism_total_hormones(**asdict(MetabolismHormones()))
        # Spatialized for all root segments in MTG...
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:
                self.transport_N(vid, **asdict(TransportCommonN()))
                self.metabolism_N(vid, **asdict(MetabolismN()))

        self.update_N(time_step=time_step, **asdict(UpdateN()))


class DiscreteVessels(CommonNitrogenModel):

    def __init__(self, g, xylem_Nm, xylem_AA, xylem_struct_mass, phloem_AA,
                 phloem_struct_mass, axial_diffusion_Nm_xylem, axial_diffusion_AA_xylem,
                 axial_diffusion_AA_phloem, **kwargs):

        # New properties' creation in MTG
        self.keywords = dict(xylem_Nm=xylem_Nm,
                            xylem_AA=xylem_AA,
                            xylem_struct_mass=xylem_struct_mass,
                            phloem_AA=phloem_AA,
                            phloem_struct_mass=phloem_struct_mass,
                            axial_diffusion_Nm_xylem=axial_diffusion_Nm_xylem,
                            axial_diffusion_AA_xylem=axial_diffusion_AA_xylem,
                            axial_diffusion_AA_phloem=axial_diffusion_AA_phloem)
        self.root_system_totals = {}

        # Properties to be accessed, pointing to g for further modifications
        self.states = """
                xylem_Nm
                xylem_AA
                xylem_struct_mass
                phloem_AA
                phloem_struct_mass
                axial_diffusion_Nm_xylem
                axial_diffusion_AA_xylem
                axial_diffusion_AA_phloem
                """.split()

        super().__init__(g, **kwargs)

    def transport_N(self, v, axial_diffusion_xylem, axial_diffusion_phloem, xylem_to_root, phloem_to_root, **kwargs):
        # RADIAL TRANSPORT

        self.transport_radial_N(v=v, model=v, xylem_to_root=xylem_to_root, phloem_to_root=phloem_to_root,
                                **kwargs)

        # AXIAL TRANSPORT

        neighbor = [self.g.parent(v)] + self.g.children(v)
        if None in neighbor:
            neighbor.remove(None)

        # Reinitialization before computing for each neighbor
        self.axial_diffusion_Nm_xylem[v] = 0.0
        self.axial_diffusion_AA_xylem[v] = 0.0
        self.axial_diffusion_AA_phloem[v] = 0.0

        for k in neighbor:
            if self.struct_mass[k] > 0:
                # MINERAL NITROGEN TRANSPORT
                self.axial_diffusion_Nm_xylem[v] += axial_diffusion_xylem * (self.xylem_Nm[k] - self.xylem_Nm[v]) * (
                                                    np.pi * (xylem_to_root * (self.radius[v] + self.radius[k]) / 2)**2)

                # AMINO ACID TRANSPORT
                self.axial_diffusion_AA_xylem[v] += axial_diffusion_xylem * (self.xylem_AA[k] - self.xylem_AA[v]) * (
                                                    np.pi * (xylem_to_root * (self.radius[v] + self.radius[k]) / 2) ** 2)

                self.axial_diffusion_AA_phloem[v] += axial_diffusion_phloem * (self.phloem_AA[k] - self.phloem_AA[v]) * (
                                                    np.pi * (xylem_to_root * (self.radius[v] + self.radius[k]) / 2) ** 2)

    def update_N(self, r_Nm_AA, r_AA_struct, r_AA_stor, xylem_to_root, phloem_to_root, time_step):
        """
        Description
        ___________

        Parameters
        __________
        :param r_Nm_struct : Nm mol consumed per mol of organic structural synthesis (adim)

        """

        # for all root segments in MTG...
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:

                # Local nitrogen concentration update
                self.update_N_local(vid, r_Nm_AA, r_AA_struct, r_AA_stor, time_step)

                # Local vessels' structural mass update
                self.xylem_struct_mass[vid] = self.struct_mass[vid] * xylem_to_root
                self.phloem_struct_mass[vid] = self.struct_mass[vid] * phloem_to_root

                # Global vessel's nitrogen pool update
                self.xylem_Nm[vid] += time_step / self.xylem_struct_mass[vid] * (
                        self.export_Nm[vid]
                        + self.diffusion_Nm_soil_xylem[vid]
                        - self.diffusion_Nm_xylem[vid]
                        + self.axial_diffusion_Nm_xylem[vid])
                self.xylem_AA[vid] += time_step / self.xylem_struct_mass[vid] * (
                        self.export_AA[vid]
                        + self.diffusion_AA_soil_xylem[vid]
                        + self.axial_diffusion_AA_xylem[vid])
                self.phloem_AA[vid] -= time_step / self.phloem_struct_mass[vid] * (
                        self.diffusion_AA_phloem[vid]
                        - self.axial_diffusion_AA_phloem[vid])

        # Update plant-level properties
        self.update_sums(time_step)
        # Vessels structural masses, defined as a fixed proportion to be abe to define concentrations
        self.xylem_total_struct_mass = self.total_struct_mass * xylem_to_root
        self.phloem_total_struct_mass = self.total_struct_mass * phloem_to_root
        # Global xylem and phloem pools for outputs
        self.xylem_total_Nm = sum(self.xylem_Nm.values())
        self.xylem_total_AA = sum(self.xylem_AA.values())
        self.phloem_total_AA = sum(self.phloem_AA.values())

    def exchanges_and_balance(self, time_step):

        """
        Description
        ___________
        Model time-step processes and balance for nitrogen to be called by simulation files.

        """

        # Computing all derivative processes
        # Global root system processes
        self.metabolism_total_hormones(**asdict(MetabolismHormones()))
        # Spatialized for all root segments in MTG...
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:
                self.transport_N(vid, **asdict(TransportAxialN()))
                self.metabolism_N(vid, **asdict(MetabolismN()))

        self.update_N(time_step=time_step, **asdict(UpdateN()))
