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
from dataclasses import dataclass

# Dataclass for initialisation and parametrization.
# For readability's sake, only units are displayed. See functions' documentation for descriptions.


# Properties' init


@dataclass
class InitCommonN:
    Nm: float = 1e-4    # mol N.g-1
    AA: float = 1e-4    # mol AA.g-1
    struct_protein: float = 0   # mol prot struct.g-1
    storage_protein: float = 0     # mol prot stor.g-1
    import_Nm: float = 0    # mol N.s-1
    export_Nm: float = 0    # mol N.s-1
    export_AA: float = 0    # mol AA.s-1
    diffusion_Nm_soil: float = 0    # mol N.s-1
    diffusion_Nm_xylem: float = 0    # mol N.s-1
    diffusion_AA_soil: float = 0  # mol AA.s-1
    diffusion_AA_phloem: float = 0    # mol AA.s-1
    AA_synthesis: float = 0    # mol AA.s-1
    struct_synthesis: float = 0    # mol struct.s-1
    storage_synthesis: float = 0    # mol stor.s-1
    AA_catabolism: float = 0    # mol AA.s-1
    storage_catabolism: float = 0    # mol stor.s-1
    xylem_Nm: float = 1e-4    # mol N.g-1
    xylem_AA: float = 1e-4    # mol AA.g-1
    xylem_struct_mass: float = 1e-3    # g
    phloem_AA: float = 1e-4    # mol AA.g-1
    phloem_struct_mass: float = 1e-3    # g
    Nm_root_shoot_xylem: float = 0    # mol N.s-1
    AA_root_shoot_xylem: float = 0    # mol AA.s-1
    AA_root_shoot_phloem: float = 0    # mol AA.s-1


@dataclass
class InitDiscreteVesselsN(InitCommonN):
    axial_diffusion_Nm_xylem: float = 0    # mol N.s-1
    axial_diffusion_AA_xylem: float = 0    # mol AA.s-1
    axial_diffusion_AA_phloem: float = 0    # mol AA.s-1


# Parameters' default value


@dataclass
class TransportCommonN:
    # architecture parameters
    xylem_to_root: float = 0.2  # adim
    phloem_to_root: float = 0.15    # adim
    parietal_differentiation: float = 1e-6   # préciser
    # kinetic parameters
    vmax_Nm_emergence: float = 1e-9     # mol N.s-1.m-2
    vmax_Nm_xylem: float = 1e-9     # mol N.s-1.m-2
    Km_Nm_root: float = 1e-4    # mol N.g-1
    Km_Nm_xylem: float = 1e-4   # mol N.g-1
    vmax_AA_xylem: float = 1e-9     # mol AA.s-1.m-2
    Km_AA_xylem: float = 1e-4   # mol AA.g-1
    diffusion_soil: float = 1e-9
    diffusion_xylem: float = 1e-8
    diffusion_phloem : float = 1e-8
    # metabolism-related parameters
    transport_C_regulation: float = 1e-2


@dataclass
class TransportAxialN(TransportCommonN):
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
class UpdateN:
    r_Nm_AA: float = 2
    r_AA_struct: float = 2
    r_AA_stor: float = 2
    xylem_to_root: float = 0.2
    phloem_to_root: float = 0.15
    time_step: int = 3600


# Nitrogen Model versions as classes. A version relates to a set of structural assumptions given in the class name.


class CommonNitrogenModel:
    def __init__(self, g, Nm, AA, struct_protein, storage_protein, import_Nm, export_Nm, export_AA, diffusion_Nm_soil,
                 diffusion_Nm_xylem, diffusion_AA_soil, diffusion_AA_phloem, AA_synthesis, struct_synthesis,
                 storage_synthesis, AA_catabolism, storage_catabolism, xylem_Nm, xylem_AA, xylem_struct_mass, phloem_AA,
                 phloem_struct_mass, Nm_root_shoot_xylem, AA_root_shoot_xylem,
                 AA_root_shoot_phloem):

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
                        diffusion_AA_soil=diffusion_AA_soil,
                        diffusion_AA_phloem=diffusion_AA_phloem,
                        AA_synthesis=AA_synthesis,
                        struct_synthesis=struct_synthesis,
                        storage_synthesis=storage_synthesis,
                        AA_catabolism=AA_catabolism,
                        storage_catabolism=storage_catabolism,
                        xylem_Nm=xylem_Nm,
                        xylem_AA=xylem_AA,
                        xylem_struct_mass=xylem_struct_mass,
                        phloem_AA=phloem_AA,
                        phloem_struct_mass=phloem_struct_mass,
                        Nm_root_shoot_xylem=Nm_root_shoot_xylem,
                        AA_root_shoot_xylem=AA_root_shoot_xylem,
                        AA_root_shoot_phloem=AA_root_shoot_phloem
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
                        diffusion_AA_soil
                        diffusion_AA_phloem
                        AA_synthesis
                        struct_synthesis
                        storage_synthesis
                        AA_catabolism
                        storage_catabolism
                        xylem_Nm
                        xylem_AA
                        xylem_struct_mass
                        phloem_AA
                        phloem_struct_mass
                        Nm_root_shoot_xylem
                        AA_root_shoot_xylem
                        AA_root_shoot_phloem
                        length
                        radius
                        struct_mass
                        C_hexose_root
                        C_hexose_reserve
                        thermal_time_since_emergence
                        """.split()

        for name in self.states:
            setattr(self, name, props[name])

        # Note : Global properties are declared as local ones, but only vertice 1 will be updated

    def metabolism_N(self, smax_AA, Km_Nm_AA, Km_C_AA, smax_struct, Km_AA_struct, smax_stor,
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
        # for all root segments in MTG...
        for vid in self.vertices:

            # if root segment emerged
            if self.struct_mass[vid] > 0:
                # amino acid synthesis
                self.AA_synthesis[vid] = smax_AA / (
                    ((1 + Km_Nm_AA) / self.Nm[vid])
                    + ((1 + Km_C_AA) / self.C_hexose_root[vid])
                )

                # Organic structure synthesis
                self.struct_synthesis[vid] = (smax_struct * self.AA[vid]
                                               / (Km_AA_struct + self.AA[vid]))

                # Organic storage synthesis (Michaelis-Menten kinetic)
                self.storage_synthesis[vid] = (smax_stor * self.AA[vid]
                                               /(Km_AA_stor + self.AA[vid]))

                # Organic storage catabolism
                Km_stor_root = Km_stor_catab * np.exp(storage_C_regulation * self.C_hexose_root[vid])
                self.storage_catabolism[vid] = cmax_stor * self.C_hexose_reserve[vid] / (
                        Km_stor_root + self.C_hexose_reserve[vid])

                # AA catabolism
                Km_stor_root = Km_AA_catab * np.exp(storage_C_regulation * self.C_hexose_root[vid])
                self.AA_catabolism[vid] = cmax_AA * self.AA[vid] / (
                        Km_stor_root + self.AA[vid])

    def transport_radial_N(self, v, model, xylem_to_root, phloem_to_root, parietal_differentiation,
                           vmax_Nm_emergence, vmax_Nm_xylem, Km_Nm_root, Km_Nm_xylem, vmax_AA_xylem,
                           Km_AA_xylem, diffusion_soil, diffusion_xylem, diffusion_phloem, transport_C_regulation):

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
        # Vmax supposed affected by root aging
        vmax_Nm_root = vmax_Nm_emergence * np.exp(
            - parietal_differentiation * self.thermal_time_since_emergence[v])

        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        self.import_Nm[v] = ((self.soil_Nm[v] * vmax_Nm_root / (self.soil_Nm[v] + Km_Nm_root))
                               * (2 * np.pi * self.radius[v] * self.length[v])
                               * (self.C_hexose_root[v] / (self.C_hexose_root[v] + transport_C_regulation)))

        # Passive radial diffusion between phloem and cortex through plasmodesmata
        self.diffusion_Nm_soil[v] = (diffusion_soil * (self.Nm[v] - self.soil_Nm[v])
                                      * (2 * np.pi * self.radius[v] * self.length[v]))

        # We define active export to xylem from root segment

        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        self.export_Nm[v] = ((self.Nm[v] * vmax_Nm_xylem / (self.Nm[v] + Km_Nm_xylem))
                                * (2 * np.pi * self.radius[v] * xylem_to_root * self.length[v])
                                * (self.C_hexose_root[v] / (
                            self.C_hexose_root[v] + transport_C_regulation)))

        # Passive radial diffusion between xylem and cortex through plasmalema
        self.diffusion_Nm_xylem[v] = (diffusion_xylem * (self.xylem_Nm[model] - self.Nm[v])
                                        * (2 * np.pi * self.radius[v] * xylem_to_root * self.length[v]))

        # AMINO ACID TRANSPORT

        # We define amino acid passive diffusion to soil
        self.diffusion_AA_soil[v] = (diffusion_soil * (self.AA[v] - self.soil_AA[v])
                                         * (2 * np.pi * self.radius[v] * phloem_to_root * self.length[v]))

        # We define active export to xylem from root segment

        # Km is defined as a constant here because xylem is global
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        self.export_AA[v] = ((self.AA[v] * vmax_AA_xylem / (self.AA[v] + Km_AA_xylem))
                                * (2 * np.pi * self.radius[v] * xylem_to_root * self.length[v])
                                * (self.C_hexose_root[v] / (
                        self.C_hexose_root[v] + transport_C_regulation)))

        # Passive radial diffusion between phloem and cortex through plasmodesmata
        self.diffusion_AA_phloem[v] = (diffusion_phloem * (self.phloem_AA[model] - self.AA[v])
                                         * (2 * np.pi * self.radius[v] * phloem_to_root * self.length[v]))

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

    def update_N_global(self, time_step):
        self.xylem_Nm[1] -= time_step * self.Nm_root_shoot_xylem[1] / self.xylem_struct_mass[1]
        self.xylem_AA[1] -= time_step * self.AA_root_shoot_xylem[1] / self.xylem_struct_mass[1]
        self.phloem_AA[1] += time_step * self.AA_root_shoot_phloem[1] / self.phloem_struct_mass[1]


class OnePoolVessels(CommonNitrogenModel):

    def __init__(self, g, **kwargs):

        self.keywords = {}
        self.states = []
        super().__init__(g, **kwargs)

    def transport_N(self, **kwargs):

        # for all root segments in MTG...
        for vid in self.vertices:

            # if root segment emerged
            if self.struct_mass[vid] > 0:
                self.transport_radial_N(vid, model=1, **kwargs)

    def update_N(self, r_Nm_AA, r_AA_struct, r_AA_stor, xylem_to_root, phloem_to_root, time_step):
        """
        Description
        ___________

        Parameters
        __________
        :param r_Nm_struct : Nm mol consumed per mol of organic structural synthesis (adim)

        """

        # Computing vessels' mass as a fraction of total segments mass
        self.xylem_struct_mass[1] = sum(self.struct_mass.values()) * xylem_to_root
        self.phloem_struct_mass[1] = sum(self.struct_mass.values()) * phloem_to_root

        # for all root segments in MTG...
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:

                # Local nitrogen concentration update
                self.update_N_local(vid, r_Nm_AA, r_AA_struct, r_AA_stor, time_step)

                # Global vessel's nitrogen pool update
                self.xylem_Nm[1] += time_step * (self.export_Nm[vid] - self.diffusion_Nm_xylem[vid]) / self.xylem_struct_mass[1]
                self.xylem_AA[1] += time_step * self.export_AA[vid] / self.xylem_struct_mass[1]
                self.phloem_AA[1] -= time_step * self.diffusion_AA_phloem[vid] / self.xylem_struct_mass[1]

        # Update plant-level properties
        self.update_N_global(time_step)


class DiscreteVessels(CommonNitrogenModel):

    def __init__(self, g, axial_diffusion_Nm_xylem, axial_diffusion_AA_xylem,
                 axial_diffusion_AA_phloem, **kwargs):

        # New properties' creation in MTG
        self.keywords = dict(axial_diffusion_Nm_xylem=axial_diffusion_Nm_xylem,
                             axial_diffusion_AA_xylem=axial_diffusion_AA_xylem,
                             axial_diffusion_AA_phloem=axial_diffusion_AA_phloem)

        # Properties to be accessed, pointing to g for further modifications
        self.states = """
                axial_diffusion_Nm_xylem
                axial_diffusion_AA_xylem
                axial_diffusion_AA_phloem
                """.split()

        super().__init__(g, **kwargs)

    def transport_N(self, axial_diffusion_xylem, axial_diffusion_phloem, xylem_to_root, phloem_to_root, **kwargs):

        # for all root segments in MTG...
        for vid in self.vertices:

            # if root segment emerged
            if self.struct_mass[vid] > 0:

                # RADIAL TRANSPORT

                self.transport_radial_N(v=vid, model=vid, xylem_to_root=xylem_to_root, phloem_to_root=phloem_to_root,
                                        **kwargs)

                # AXIAL TRANSPORT

                neighbor = [self.g.parent(vid)] + self.g.children(vid)
                if None in neighbor:
                    neighbor.remove(None)

                # Reinitialization before computing for each neighbor
                self.axial_diffusion_Nm_xylem[vid] = 0.0
                self.axial_diffusion_AA_xylem[vid] = 0.0
                self.axial_diffusion_AA_phloem[vid] = 0.0

                for k in neighbor:
                    if self.struct_mass[k] > 0:
                        # MINERAL NITROGEN TRANSPORT
                        self.axial_diffusion_Nm_xylem[vid] += axial_diffusion_xylem * (self.xylem_Nm[k] - self.xylem_Nm[vid]) * (
                                                            np.pi * (xylem_to_root * (self.radius[vid] + self.radius[k]) / 2)**2)

                        # AMINO ACID TRANSPORT
                        self.axial_diffusion_AA_xylem[vid] += axial_diffusion_xylem * (self.xylem_AA[k] - self.xylem_AA[vid]) * (
                                                            np.pi * (xylem_to_root * (self.radius[vid] + self.radius[k]) / 2) ** 2)

                        self.axial_diffusion_AA_phloem[vid] += axial_diffusion_phloem * (self.phloem_AA[k] - self.phloem_AA[vid]) * (
                                                            np.pi * (xylem_to_root * (self.radius[vid] + self.radius[k]) / 2) ** 2)

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
                        - self.diffusion_Nm_xylem[vid]
                        + self.axial_diffusion_Nm_xylem[vid])
                self.xylem_AA[vid] += time_step / self.xylem_struct_mass[vid] * (
                        self.export_AA[vid]
                        + self.axial_diffusion_AA_xylem[vid])
                self.phloem_AA[vid] -= time_step / self.xylem_struct_mass[vid] * (
                        self.diffusion_AA_phloem[vid]
                        - self.axial_diffusion_AA_phloem[vid])

        # Update plant-level properties
        self.update_N_global(time_step)

