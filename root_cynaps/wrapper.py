import pickle
from functools import partial

class ModelWrapper:
    @property
    def documentation(self):
        """
        Documentation of the RootCyNAPS parameters
        :return: documentation text
        """
        return dict(zip((name, value) for name, value in self.root_nitrogen.__dataclassfields__ if
                        value.metadata["variable_type"] == "state_variable"))

    @property
    def inputs(self):
        return

    def scenario(self, **kwargs):
        """
        Method
        """
        for model in self.models:
            for changed_parameter, value in kwargs:
                if changed_parameter in model.__dict__:
                    setattr(model, changed_parameter, value)

    def link_around_mtg(self, translator={}):
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
        L = len(self.models)
        for receiver_index in range(L):
            for applier_index in range(L):
                if receiver_index != applier_index:
                    receiver = self.models[receiver_index]
                    applier = self.models[applier_index]
                    linker = translator[receiver_index][applier_index]
                    for name, value in linker.items():
                        setattr(receiver, name, partial(self.single_linker, dict(source=applier, d=value)))

    def single_linker(self, source, d):
        if len(d.keys()) > 1:
            return sum([getattr(source, n) * v for n, v in d.items()])
        else:
            return getattr(source, d.keys()[0])

    def translator_utility(self):
        L = len(self.models)
        translator = [[{} for k in range(L)]]
        for receiver_model in range(L):
            inputs = dict(zip((name, value) for name, value in self.models[receiver_model].__dataclassfields__ if value.metadata["variable_type"] == "input"))
            needed_models = list(set([value.metadata["by"] for value in inputs.values()]))
            for name in needed_models:
                print([(self.models.index(k) + 1, k) for k in self.models])
                which = int(input(f"Which is {name}? : ")) - 1
                needed_inputs = [name for name, value in inputs if value.metadata["by"] == name]
                for var in needed_inputs:
                    print(var, inputs[var])
                    available = [(name, value) for name, value in self.models[which].__dataclassfields__ if value.metadata["variable_type"] == "state_variable"]
                    print(available)
                    selected = input("Enter target names * unit conversion factor. Separate by ;       -> ").split(";")
                    com_dict = {}
                    for expression in selected.replace(" ", ""):
                        if "*" in expression:
                            l = expression.split("*")
                            com_dict[l[0]] = float(l[1])
                        else:
                            com_dict[expression] = 1.
                    translator[receiver_model][which][var] = com_dict

        with open("translator.pckl", "wb") as f:
            pickle.dump(translator, f)
