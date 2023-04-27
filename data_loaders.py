import yaml
import pandas as pd
import numpy as np
from typing import Tuple, Dict

def load_config(filename: str) -> Dict:
    return yaml.safe_load(open())

def load_data(prefs_file: str, exper_file: str) -> Tuple[pd.DataFrame, np.array, np.array]:
    prefs = pd.read_csv(prefs_file, index_col="slot")
    slot_names = prefs.index.to_list()
    person_names = prefs.columns.to_list()[0:]
    prefs_normed = prefs / prefs.sum(axis=0) * len(slot_names)
    # convert to numpy for model

    exp_df = pd.read_csv(exper_file)
    exp_dict = {k: v for k, v in zip(exp_df.columns, exp_df.values[0])}
    exp_indicator = np.array([int(exp_dict[n] > CONF["experience_months"]) for n in person_names])

    return (prefs_normed, prefs_normed.to_numpy(), exp_indicator)