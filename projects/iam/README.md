# Overprojection of Carbon Capture and Storage in Global Decarbonization Scenarios
- Lindsey Gulden
- lindsey at legupdata dot com
- April 2026

## What is in this directory?
Files included here are python scripts, jupyter notebooks, and yaml configuration files used to implement research underpinning the following paper: [Overprojection of Carbon Capture and Storage in Global Decarbonization Scenarios](https://docs.google.com/document/d/1TQpmiwgphxeU774to0shgKQuwI34sTGSdsfw_6J1YfM/edit?usp=sharing).
* Python code for implementing ESLiM simulations is found in [eslim.py](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/eslim.py).
* Python code for the command-line script that implements LHS-sampled uncertainty analysis and Saltelli Sensitivity analysis of ESLiM can be found in [sensitivity.py](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/sensitivity.py). To use, at the command line, type the following:
    ``` python3 [path/to/this/file] --config [path/to/sensitivity_config.yml] ```
* Configuration YAML files containing details for individual ESLiM model runs (e.g., the IEA/IPCC-default parameter values/specifications for ESLiM, found in [eslim_baseine_config.yml](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/config/eslim_baseline_config.yml)) as well as LHS Uncertainty analysis ([lhs_config.yml](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/config/lhs_config.yml)) and Saltelli global sensitivity analysis ([sensitivity_config.yml](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/config/sensitivity_config.yml)) can be found in the [config](https://github.com/lindseygulden/leg-up/tree/main/projects/iam/config) directory
* Data processing, analysis, and figure generation for all figures in the manuscript can be found in the [notebooks](https://github.com/lindseygulden/leg-up/tree/main/projects/iam/notebooks) directory:
    * Figures based on IEA outlook data and the AR6 IPCC WGIII data are in [figs_and_analysis_IEA_and_IPCC_data.ipynb](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/notebooks/figs_and_analysis_IEA_and_IPCC_data.ipynb). Each of the three notebooks requires the user to modify the local path that points to the [data_and_config_locations.yml](https://github.com/lindseygulden/leg-up/tree/main/projects/iam/config/data_and_config_locations.yml) configuration file (which can be found in the 'Setup' block at the top of each notebook).
    * Analysis and figures for the ESLiM named model simulations (e.g., IEA/IPCC default, midpoint simulations, etc.), uncertainty analysis, and sensitivitiy analysis can be found in [figs_and_analysis_eslim.ipynb](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/notebooks/figs_and_analysis_eslim.ipynb)
    * Several figures to explore the behavior of a nested-logit choice function, as is implemented in ESLiM can be found in [figs_logit_choice.ipynb](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/notebooks/figs_logit_choice.ipynb). Some figures are included in the manuscript linked above; others are bonus figs.

## Additional notes
Files here depend on several helper functions in the [utils](https://github.com/lindseygulden/leg-up/tree/main/utils) folder

