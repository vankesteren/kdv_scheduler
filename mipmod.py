# KDV model object, open source version
# last edited 20230428 by @vankesteren
import mip
import yaml
import pandas as pd
import numpy as np
from datetime import datetime as dt
import os

class KDVModel:
    def __init__(self, config_file: str, preferences_file: str, experiences_file: str):
        # instantiate gurobi model
        self.model = mip.Model("KDVModel", solver_name="CBC")

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
        self.prefs_normed = self.prefs / self.prefs.sum(axis=0) #* len(self.slot_names)
        self.prefs_np = self.prefs_normed.to_numpy()

    def load_experiences(self, experiences_file: str) -> None:
        """Load the experiences file and store it in the object"""
        exp_df = pd.read_csv(experiences_file)
        exp_dict = {k: v for k, v in zip(exp_df.columns, exp_df.values[0])}
        self.exp_indicator = np.array([int(exp_dict[n] > self.config["experience_months"]) for n in self.person_names])

    # Model setup methods
    def set_variables(self) -> None:
        self.var_names = np.array([[f"{n}_{s}" for n in self.person_names] for s in self.slot_names])
        self.assignments = self.model.add_var_tensor(self.prefs.shape, var_type=mip.BINARY, name="assignments")
        self.slots_per_person = self.assignments.sum(axis=0)
        self.persons_per_slot = self.assignments.sum(axis=1)
        self.experienced_per_slot = self.assignments @ self.exp_indicator

    def set_objective(self) -> None:
        self.discrepancy = self.assignments - self.prefs_np
        self.model.objective = mip.minimize(mip.xsum(d.item() for d in np.nditer(self.discrepancy, flags=["refs_ok"])))
    
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
                self.model += self.assignments[idx] == 0, f"unavail_{self.var_names[idx]}"

    def constr_slots_per_person(self) -> None:
        """Ensure that persons have at most slots_per_person_max slots assigned to them."""
        for i in range(len(self.person_names)):
            self.model += self.slots_per_person[i] <= self.config["slots_per_person_max"], f"maxslots_{self.person_names[i]}"
            
    def constr_persons_per_slot(self) -> None:
        """Ensure at least persons_per_slot_min and at most persons_per_slot_max persons per slot."""
        for i in range(len(self.slot_names)):
            self.model += self.persons_per_slot[i] >= self.config["persons_per_slot_min"], f"minpersons_{self.slot_names[i]}"
            self.model += self.persons_per_slot[i] <= self.config["persons_per_slot_max"], f"maxpersons_{self.slot_names[i]}"
            
    def constr_experienced_persons(self) -> None:
        """Ensure at least min_experienced_persons of experienced persons per slot"""
        for i in range(len(self.slot_names)):
            self.model += self.experienced_per_slot[i] >= self.config["min_experienced_persons"], f"expperson_{self.slot_names[i]}"

    # Model convenience methods
    def optimize(self, *args, **kwargs) -> None: 
        self.opt_status = self.model.optimize(*args, **kwargs)
        self.curtime = dt.today()

    def update(self, *args, **kwargs) -> None: 
        self.model.update(*args, **kwargs)

    def converged(self) -> bool:
        return self.opt_status == mip.OptimizationStatus.OPTIMAL or self.opt_status == mip.OptimizationStatus.FEASIBLE
    
    # Model output methods
    def slot_schedule(self) -> pd.DataFrame:
        assert self.converged()
        res = self.prefs.copy()
        res[:] = np.reshape([v.item().x for v in np.nditer(self.assignments, flags=["refs_ok"])], self.assignments.shape) 
        return res

    def slot_desirability(self) -> pd.DataFrame:
        """if everyone would fill out the same number everywhere, this would be 1"""
        return pd.DataFrame({"desirability": self.prefs_normed.sum(axis=1) / len(self.person_names) * len(self.slot_names)})
    
    def person_flexibility(self) -> pd.DataFrame:
        """0.1 = only one slot available, 1.0 = all available, no preference"""
        return pd.DataFrame({"flexibility": 1 / self.prefs_normed.max(axis=0)})
    
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
