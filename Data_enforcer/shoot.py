import os
from dataclasses import dataclass, field, fields
import pandas as pd


@dataclass
class ShootModel:
    # Just doc
    Total_Transpiration: dict = field(default=0., metadata=dict(unit="mol.h-1", unit_comment="of water", description="Total water transpiration reported at collar", value_comment="", references="Barillot et al., 2016; Gauthier et al., 2020", variable_type="state_variable", by="model_shoot", state_variable_type="extensive", edit_by="user"))
    Unloading_Amino_Acids: dict = field(default=0., metadata=dict(unit="mol.h-1", unit_comment="of amino acids", description="Unloading of amino acids by shoot in phloem", value_comment="", references="Barillot et al., 2016; Gauthier et al., 2020", variable_type="state_variable", by="model_shoot", state_variable_type="extensive", edit_by="user"))
    Export_cytokinins: dict = field(default=0., metadata=dict(unit="mol.h-1", unit_comment="of water", description="export of root produced cytokinins to shoot through xylem", value_comment="", references="Barillot et al., 2016; Gauthier et al., 2020", variable_type="state_variable", by="model_shoot", state_variable_type="extensive", edit_by="user"))
    Unloading_Sucrose: dict = field(default=0., metadata=dict(unit="mol.h-1", unit_comment="of water", description="Unloading of sucrose by shoot in phloem", value_comment="", references="Barillot et al., 2016; Gauthier et al., 2020", variable_type="state_variable", by="model_shoot", state_variable_type="extensive", edit_by="user"))

    def __init__(self, g):

        self.g = g
        self.dataset = pd.read_csv(os.path.dirname(__file__) + "/inputs/cnwheat_outputs.csv", sep=";")
        self.dataset = self.dataset.set_index("t")

        props = self.g.properties()
        for name in self.dataset.columns:
            props.setdefault(name, {})
            props[name][1] = self.dataset[name][0]
            setattr(self, name, props[name])

    def transportW(self, time):
        # At each time step, we set the transpiration value from the csv file
        self.Total_Transpiration[1] = self.dataset["Total_Transpiration"][time] * 1e-3  # mmol.s-1

    def transportN(self, time):
        self.Unloading_Amino_Acids[1] = self.dataset["Unloading_Amino_Acids"][time] * 1e-6  # micromol.h-1
        self.Export_cytokinins[1] = self.dataset["Export_cytokinins"][time]

    def transportC(self, time):
        self.Unloading_Sucrose[1] = self.dataset["Unloading_Sucrose"][time] * 1e-6 / 3600  # micromol.h-1 inputs

    def run_exchanges_and_balance(self, time):
        # Water flow first for advection computation
        self.transportW(time)
        self.transportN(time)
        self.transportC(time)
