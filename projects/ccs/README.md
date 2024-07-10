
# README
Files written in June 2024, to support rigorous evaluation of the financial viability and emissions-reduction potential of carbon sequestration and storage as well as rooftop solar. Multiple work products resulted from this work:
(1) Paper to be submitted to the Climate Resilience and Climate Justice special issue on carbon removal.
(2) Editorial to be submitted regarding transparency in research that founds public policy
(3) Blog post pointing out the impact of unrealistic assumptions on Rhodium's results
The work grew out of initial analysis of Rhodium Group's May 2024 projections for future adoption of CCS for industrial decarbonization in the United States (See report at [Expanding the Industrial Decarbonization Toolkit](https://rhg.com/research/expanding-the-industrial-decarbonization-toolkit/).) Included is code to build a companion, 'doppelganger' simulator that mirrors what is used to generate the CCS capacity projections reported in the May 2024 Rhodium Group report as well as simulation capability for evaluating the GHG-reduction potential and community-investment impact of rooftop solar arrays.

## What's here?
* Configuration files: to enable tracking of data used and facilitate changing of settings, data location, etc.
* Data-processing scripts: for querying APIs and assembling data from various public sources
* Class definitions: to set up a parent 'Project' class that tracks a project's cash flow and NPV as well as to child classes (RooftopSolarProject and CCSProject)
* Functions/scripts for running probabilistic and ensemble simulations using the RooftopSolarProject instances and the CCSProject instances
* Jupyter notebooks for generating figures and doing basic data processing, reporting

### Configuration information stored in YAML files
Configuration files house paths to data files, modeling parameters, etc.
[CCS cost calculation configuration yaml](https://github.com/lindseygulden/leg-up/blob/main/config/ccs/ccs_cost_info.yml)

[CCS RHG Emissions Scenarios simulation configuration yaml](https://github.com/lindseygulden/leg-up/blob/main/config/ccs/rhg_scenarios.yml)

[‘Real world’ ensemble simulation configuration information for our cash flow model ('UES') -- in yaml](https://github.com/lindseygulden/leg-up/blob/main/config/ccs/real_world_scenarios.yml)

[Configuration details for use in the get_pvwatts_solar_data.py click/command-line script that queries NREL's PVWatts API](https://github.com/lindseygulden/leg-up/blob/main/config/ccs/pvwatts.yml)

[Locations of figures and data for input to the Jupyter notebook in which the figures are generated](https://github.com/lindseygulden/leg-up/blob/main/config/ccs/fig_data_locs.yml)

[Paths to data, info used to set up/run the rooftop solar project cases](https://github.com/lindseygulden/leg-up/blob/main/config/ccs/solar_comparison_case.yml)

### Python Objects used at core of simulations
We define a unit-economics simulator base class and CCS subclass
[Abstract base class (Project) for unit-economics project-economics simulation](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/project.py)

[CCSProject: Subclass for Project parent class that represents a Carbon Capture and Sequestration project](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/ccs_project.py)
Defines a Carbon Capture and Storage project; computes key characteristics based on inputs.

[RooftopSolarProject: Subclass for Project parent class that represents a Rooftop Photovolataic Array project](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/rooftop_solar_project.py)
Defines a rooftop solar array project; computes key characteristics based on inputs.

### Data-processing/querying scripts used for generating scenarios and financial/cost assumptions
[Command-line script used to adjust daily Brent prices to 2023 dollars and compute the rolling-window 365-day average price for Brent](https://github.com/lindseygulden/leg-up/blob/main/data/processing_scripts/adjust_prices.py)

[Command-line script to assign approximate lat/lon locations to ammonia facilities (based on text city and state data)](https://github.com/lindseygulden/leg-up/blob/main/data/processing_scripts/ammonia_plant_location.py)

[Command line script used to make the data frame containing the triangular distribution parameters for breakeven prices for new and existing wells](https://github.com/lindseygulden/leg-up/blob/main/data/processing_scripts/build_breakeven_dataset.py)

[Command-line script to extract US cement production facilities from global database](https://github.com/lindseygulden/leg-up/blob/main/data/processing_scripts/cement.py)

[Command-line script to query the PVWatts v8 API for various lat/lons, azimuths, and roof tilts](https://github.com/lindseygulden/leg-up/blob/main/data/processing_scripts/get_pvwatts_solar_data.py)

[Bespoke script for compiling solar production, electricity rate, carbon intensity and PV array installation cost](https://github.com/lindseygulden/leg-up/blob/main/data/processing_scripts/assemble_rooftop_solar_data.py)

### Functions for running scenarios and probabilistic scenarios (eventually should be pulled into object oriented framework, above)
[Function to assemble and run the Unit-Economics Simulator with three Rhodium Group (RHG) Emissions Pathways/Scenarios, specified in a configuration yaml](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/rhg_scenarios.py)

[Function that generates ensemble simluations of the Unit-Economics Simulator](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/ensembles.py)
Uses randomly sampled oil prices, capture/transport/storage costs, and oil breakeven price

[Functions to extract, process, and update costs for carbon capture, transport, and storage](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/ccs_costs.py)

### Running the whole enchilada
[Script to running cost and location computation, RHG ensembles, and the UES ‘realistic’ ensembles](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/run_ccs_analysis.py)

[Script to run ensemble of rooftop photovoltaic array simulations for cash flow and emissions reduction](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/run_rooftop_solar.py)

### Figures and tables for presenting results
[Jupyter notebook for constructing figures and post-processing](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/notebooks/figures_and_postprocessing_ccs_profitability_analysis.ipynb). Note that configuration files (i.e., location of data to be used for figures/tables) is found in
[Jupyter notebook for comparing 2024 Rhodium projections to similar CCS projections made in 2020](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/notebooks/rhg_ccs.ipynb)
