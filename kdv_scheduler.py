# KDV scheduler script using gurobi
# last edited 20230427 by @vankesteren
from kdv_model import KDVModel

# Instantiate a KDV model using the input files
mod = KDVModel("input_files/config.yml", "input_files/preferences.csv", "input_files/experience.csv")

# Then, compute the optimal slot schedule
mod.optimize()

# inspect output
mod.slot_schedule()
mod.slot_desirability()
mod.person_flexibility()

# store output
mod.save_output()