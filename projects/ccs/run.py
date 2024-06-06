"""Hack script to run various RHG CCS economics evaluation"""

import pandas as pd

from projects.ccs.ccs_costs import costs
from projects.ccs.ensembles import ues_ensemble
from projects.ccs.rhg_scenarios import rhg
from utils.io import yaml_to_dict

RHG_SCENARIO_CONFIG = "/Users/lindseygulden/dev/leg-up/config/rhg_scenarios.yml"

REALWORLD_CONFIG = "/Users/lindseygulden/dev/leg-up/config/real_world_scenarios.yml"

COSTS_CONFIG = "/Users/lindseygulden/dev/leg-up/config/ccs_cost_info.yml"
# compute costs for all industries
all_industries_gdf, costs_df = costs(
    COSTS_CONFIG,
)

# Use UES to run the three rhodium scenarios (script writes out files to locations in config file)
_, _, _ = rhg(RHG_SCENARIO_CONFIG)

realworld = yaml_to_dict(REALWORLD_CONFIG)
brent_df = pd.read_csv(realworld["path_to_brent_data"], index_col=[0])
costs_df = pd.read_csv(realworld["path_to_cost_data"], index_col=[0])
breakeven_df = pd.read_csv(realworld["path_to_breakeven_data"], index_col=[0])
scenario_df = ues_ensemble(realworld, costs_df, brent_df, breakeven_df)
