# A critical assessment of the treatement of CCS in the IEA and IPCC energy-system models
- Lindsey Gulden
- lindsey at legupdata dot com
- January 2026

## What is in this directory?
Files included here are python scripts, jupyter notebooks, and yaml configuration files used to implement research underpinning the following paper: [A critical assessment of the IPCC and IEA’s projections for carbon capture and storage](https://docs.google.com/document/d/1mwiSPu30LqLGQT-hkVah-VZzFI5dpKgsWUGA0rdgobI/edit?usp=sharing).
* Python code for implementing ESLiM simulations is found in [eslim.py](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/eslim.py).
* Python code for the command-line script that implements LHS-sampled uncertainty analysis and Saltelli Sensitivity analysis of ESLiM can be found in [sensitivity.py](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/sensitivity.py). To use:
    > python3 [path/to/this/file] --config [path/to/sensitivity_config.yml]
* Configuration YAML files containing details for individual ESLiM model runs (e.g., the IEA/IPCC-default parameter values/specifications for ESLiM, found in [eslim_baseine_config.yml](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/config/eslim_baseline_config.yml)) as well as LHS Uncertainty analysis ([lhs_config.yml](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/config/lhs_config.yml)) and Saltelli global sensitivity analysis ([sensitivity_config.yml](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/config/sensitivity_config.yml)) can be found in the [config](https://github.com/lindseygulden/leg-up/tree/main/projects/iam/config) directory
* Data processing, analysis, and figure generation for all figures in the manuscript can be found in the [notebooks](https://github.com/lindseygulden/leg-up/tree/main/projects/iam/notebooks) directory:
    * Figures based on IEA outlook data and the AR6 IPCC WGIII data are in [figs_and_analysis_IEA_and_IPCC_data.ipynb](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/notebooks/figs_and_analysis_IEA_and_IPCC_data.ipynb)
    * Analysis and figures for the ESLiM named model simulations (e.g., IEA/IPCC default, midpoint simulations, etc.), uncertainty analysis, and sensitivitiy analysis can be found in [figs_and_analysis_eslim.ipynb](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/notebooks/figs_and_analysis_eslim.ipynb)
    * Several figures to explore the behavior of a nested-logit choice function, as is implemented in ESLiM can be found in [figs_logit_choice.ipynb](https://github.com/lindseygulden/leg-up/blob/main/projects/iam/notebooks/figs_logit_choice.ipynb). Some figures are included in the manuscript linked above; others are bonus figs.

