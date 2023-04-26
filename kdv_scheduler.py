# KDV scheduler script using gurobi
# last edited 20230426 by @vankesteren
import gurobipy as gb
import numpy as np
import pandas as pd
from datetime import datetime as dt
import os

# File locations
PREFS_FILE = "input_files/preferences.csv"
EXPER_FILE = "input_files/experience.csv"
OUT_FOLDER = "output_files"

# Constraint settings
SLOTS_PER_PERSON_MAX = 2 # maximum number of slots per person
PERSONS_PER_SLOT_MIN = 2 # minimum number of persons per slot
PERSONS_PER_SLOT_MAX = 2 # maximum number of persons per slot
EXPERIENCE_MONTHS = 6 # number of months of experience after which a person is experienced
MIN_EXPERIENCED_PERSONS = 1 # minimum number of experienced persons per slot

# Preferences data processing
prefs = pd.read_csv(PREFS_FILE, index_col="slot")
slot_names = prefs.index.to_list()
person_names = prefs.columns.to_list()[0:]
prefs_normed = prefs / prefs.sum(axis=0) * len(slot_names)

# convert to numpy for model
prefs_np = prefs_normed.to_numpy()

# Experience data processing
exp_df = pd.read_csv(EXPER_FILE)
exp_dict = {k: v for k, v in zip(exp_df.columns, exp_df.values[0])}
exp_indicator = np.array([int(exp_dict[n] > EXPERIENCE_MONTHS) for n in person_names])

# instantiate model
kdv_mod = gb.Model("kdv_scheduler_model")

# create assignment matrix as binary gurobi variables
var_names = np.array([[f"{n} - {s}" for n in person_names] for s in slot_names])
assignments = kdv_mod.addMVar(prefs.shape, vtype="B", name=var_names)
discrepancy = assignments - prefs_np

# add objective
kdv_mod.setObjective(gb.quicksum(gb.quicksum(discrepancy * discrepancy)))

# add constraints
# persons cannot be assigned to unavailable slots
for row in range(len(slot_names)):
    for col in range(len(person_names)):
        if prefs_np[row, col] == 0:
            kdv_mod.addConstr(assignments[row, col] == 0, f"unavail_{var_names[row, col]}")

# persons should have at most SLOTS_MAX assigned slots
total_per_person = assignments.sum(axis=0)
for i in range(len(person_names)):
    kdv_mod.addConstr(total_per_person[i] <= SLOTS_PER_PERSON_MAX, name=f"Total {person_names[i]}")

# in each slot, there needs to be at least 2 people, of which 1 is experienced
total_per_slot = assignments.sum(axis = 1)
experienced_per_slot = assignments @ exp_indicator
for i in range(len(slot_names)):
    kdv_mod.addConstr(total_per_slot[i] >= PERSONS_PER_SLOT_MIN, name=f"minperson_{slot_names[i]}")
    kdv_mod.addConstr(total_per_slot[i] <= PERSONS_PER_SLOT_MAX, name=f"maxperson_{slot_names[i]}")
    kdv_mod.addConstr(experienced_per_slot[i] >= MIN_EXPERIENCED_PERSONS, name=f"expperson_{slot_names[i]}")

# optimize!
kdv_mod.optimize()

if kdv_mod.solcount == 0:
    exit(1)

# Create output file
# Create subfolder for this run
curtime = dt.today().strftime('%Y%m%d_%H%M%S')
folder_name = os.path.join("output_files", curtime)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Compute slot desirability
# if everyone would fill out the same number everywhere, this would be 1
slot_desirability = prefs_normed.sum(1) / len(person_names)
pd.DataFrame({"desirability": slot_desirability}).to_csv(os.path.join(folder_name, "desirability.csv"))

# Compute person flexibility
# 0.1 = only one slot available, 1.0 = all available, no preference
person_flexibility = 1/prefs_normed.max(0)
pd.DataFrame({"flexibility": person_flexibility}).to_csv(os.path.join(folder_name, "flexibility.csv"))

# Create person-slot assignment file
out = prefs.copy()
out[:] = assignments.X
out.to_csv(os.path.join(folder_name, "assignments.csv"))