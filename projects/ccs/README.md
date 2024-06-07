
# README
Files written in early June, 2024 for the Science Roundtable.
Lindsey Gulden

## What's here?
These are files used to support analysis and extension of Rhodium Group's 2024 projections for future adoption of CCS for industrial decarbonization in the United States. Included is code to build a companion, 'doppelganger' simulator that mirrors what is used to generate the CCS capacity projections reported in the May 2024 Rhodium Group report [Expanding the Industrial Decarbonization Toolkit](https://rhg.com/research/expanding-the-industrial-decarbonization-toolkit/)

### Configuration information stored in YAML files
Configuration files house paths to data files, modeling parameters, etc.
[CCS cost calculation configuration yaml](https://github.com/lindseygulden/leg-up/blob/main/config/ccs/ccs_cost_info.yml)

[CCS RHG Emissions Scenarios simulation configuration yaml](https://github.com/lindseygulden/leg-up/blob/main/config/ccs/rhg_scenarios.yml)

[UES ‘real world’ ensemble simulation configuration information yaml](https://github.com/lindseygulden/leg-up/blob/main/config/ccs/real_world_scenarios.yml)

### Python Objects used at core of simulations
We define a unit-economics simulator base class and CCS subclass
[Abstract base class (Project) for unit-economics project-economics simulation](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/project.py)

[CCSProject: Subclass for Project parent class that represents a Carbon Capture and Sequestration project](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/ccs_project.py)
Defines a Carbon Capture and Storage project; computes key characteristics based on inputs.

### Command line scripts to process data used for generating scenarios and financial/cost assumptions
[Command-line script used to adjust daily Brent prices to 2023 dollars and compute the rolling-window 365-day average price for Brent](https://github.com/lindseygulden/leg-up/blob/main/data/processing_scripts/adjust_prices.py)

[Command-line script to assign approximate lat/lon locations to ammonia facilities (based on text city and state data)](https://github.com/lindseygulden/leg-up/blob/main/data/processing_scripts/ammonia_plant_location.py

[Command line script used to make the data frame containing the triangular distribution parameters for breakeven prices for new and existing wells](https://github.com/lindseygulden/leg-up/blob/main/data/processing_scripts/build_breakeven_dataset.py

[Command-line script to extract US cement production facilities from global database](https://github.com/lindseygulden/leg-up/blob/main/data/processing_scripts/cement.py)

### Functions for running scenarios and probabilistic scenarios (eventually should be pulled into object oriented framework, above)
[Function to assemble and run the Unit-Economics Simulator with three Rhodium Group (RHG) Emissions Pathways/Scenarios, specified in a configuration yaml](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/rhg_scenarios.py)

[Function that generates ensemble simluations of the Unit-Economics Simulator](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/ensembles.py)
Uses randomly sampled oil prices, capture/transport/storage costs, and oil breakeven price

[Functions to extract, process, and update costs for carbon capture, transport, and storage](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/ccs_costs.py)

### Running the whole enchilada
[Script to running cost and location computation, RHG ensembles, and the UES ‘realistic’ ensembles](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/run_ccs_analysis.py)

### Figures and tables for presenting results
[Jupyter notbook for constructing figures and post-processing](https://github.com/lindseygulden/leg-up/blob/main/projects/ccs/notebooks/figures_and_postprocessing_ccs_profitability_analysis.ipynb)
