# KDV model object
# last edited 20230427 by @vankesteren
import gurobipy as gb
import yaml
import pandas as pd
import numpy as np
from datetime import datetime as dt
import os
from typing import Tuple, Dict

class KDVModel:
    def __init__(self, config_file: str, preferences_file: str, experiences_file: str):
        # instantiate gurobi model
        self.model = gb.Model("KDVModel")

        # load data
        self.load_config(config_file)
        self.load_preferences(preferences_file)
        self.load_experiences(experiences_file)

        # set up the model
        self.set_variables()
        self.set_objective()
        self.set_constraints()

    # Data Loading methods
    def load_config(self, config_file: str) -> None:
        """Load the configuration file and store it in the object"""
        self.config = yaml.safe_load(open(config_file))

    def load_preferences(self, preferences_file: str) -> None:
        """Load the preferences file and store it in the object"""
        self.prefs = pd.read_csv(preferences_file, index_col="slot")
        self.slot_names = self.prefs.index.to_list()
        self.person_names = self.prefs.columns.to_list()
        self.prefs_normed = self.prefs / self.prefs.sum(axis=0) * len(self.slot_names)
        self.prefs_np = self.prefs_normed.to_numpy()

    def load_experiences(self, experiences_file: str) -> None:
        """Load the experiences file and store it in the object"""
        exp_df = pd.read_csv(experiences_file)
        exp_dict = {k: v for k, v in zip(exp_df.columns, exp_df.values[0])}
        self.exp_indicator = np.array([int(exp_dict[n] > self.config["experience_months"]) for n in self.person_names])

    # Model setup methods
    def set_variables(self) -> None:
        self.var_names = np.array([[f"{n} - {s}" for n in self.person_names] for s in self.slot_names])
        self.assignments = self.model.addMVar(self.prefs.shape, vtype="B", name=self.var_names)
        self.slots_per_person = self.assignments.sum(axis=0)
        self.persons_per_slot = self.assignments.sum(axis=1)
        self.experienced_per_slot = self.assignments @ self.exp_indicator

    def set_objective(self) -> None:
        self.discrepancy = self.assignments - self.prefs_np
        self.model.setObjective(gb.quicksum(gb.quicksum(self.discrepancy * self.discrepancy)))
    
    def set_constraints(self) -> None:
        self.constr_no_unavailable_slots()
        self.constr_slots_per_person()
        self.constr_persons_per_slot()
        self.constr_experienced_persons()

    # Constraint methods
    def constr_no_unavailable_slots(self) -> None:
        """Make sure that persons cannot be assigned to slots they are not available for."""
        for idx, val in np.ndenumerate(self.prefs_np):
            if val == 0:
                self.model.addConstr(self.assignments[idx] == 0, f"unavail_{self.var_names[idx]}")

    def constr_slots_per_person(self) -> None:
        """Ensure that persons have at most slots_per_person_max slots assigned to them."""
        for i in range(len(self.person_names)):
            self.model.addConstr(self.slots_per_person[i] <= self.config["slots_per_person_max"], name=f"maxslots_{self.person_names[i]}")
            
    def constr_persons_per_slot(self) -> None:
        """Ensure at least persons_per_slot_min and at most persons_per_slot_max persons per slot."""
        for i in range(len(self.slot_names)):
            self.model.addConstr(self.persons_per_slot[i] >= self.config["persons_per_slot_min"], name=f"minpersons_{self.slot_names[i]}")
            self.model.addConstr(self.persons_per_slot[i] <= self.config["persons_per_slot_max"], name=f"maxpersons_{self.slot_names[i]}")
            
    def constr_experienced_persons(self) -> None:
        """Ensure at least min_experienced_persons of experienced persons per slot"""
        for i in range(len(self.slot_names)):
            self.model.addConstr(self.experienced_per_slot[i] >= self.config["min_experienced_persons"], name=f"expperson_{self.slot_names[i]}")

    # Model convenience methods
    def optimize(self, *args, **kwargs) -> None: 
        self.model.optimize(*args, **kwargs)
        self.curtime = dt.today()

    def update(self, *args, **kwargs) -> None: 
        self.model.update(*args, **kwargs)

    def converged(self) -> bool:
        return self.model.solcount > 0
    
    # Model output methods
    def slot_schedule(self) -> pd.DataFrame:
        assert self.converged()
        res = self.prefs.copy()
        res[:] = self.assignments.X
        return res

    def slot_desirability(self) -> pd.DataFrame:
        """if everyone would fill out the same number everywhere, this would be 1"""
        return pd.DataFrame({"desirability": self.prefs_normed.sum(1) / len(self.person_names)})
    
    def person_flexibility(self) -> pd.DataFrame:
        """0.1 = only one slot available, 1.0 = all available, no preference"""
        return pd.DataFrame({"flexibility": 1 / self.prefs_normed.max(0)})

    # Storing model output
    def save_output(self) -> str:
        """Stores model output and returns the folder name of the stored output"""
        assert self.converged()

        # Create subfolder for this run
        folder_name = os.path.join(self.config["output_folder"], self.curtime.strftime('%Y%m%d_%H%M%S'))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Store files in folder
        self.slot_schedule().to_csv(os.path.join(folder_name, "assignments.csv"))
        self.slot_desirability().to_csv(os.path.join(folder_name, "desirability.csv"))
        self.person_flexibility().to_csv(os.path.join(folder_name, "flexibility.csv"))

        return folder_name
