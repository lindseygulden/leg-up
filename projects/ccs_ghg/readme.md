# What's in this directory?

Code in this directory computes both industrial process GHG emissions and GHG emissions from CCS processes themselves. Computations cover Scope 1, Scope 2, and selected Scope 3 emissions for both CO2 and CH4 (see [Emissions scope](#emissions-scope) for details).


## ccs_ghg

Compute greenhouse gas (GHG) emissions for industrial carbon capture and storage (CCS). Models emissions for different industries, including emissions resulting from the non-CCS project and emissions directly relating to CCS itself (capture, transport, storage and combustion of oil in the case of enhanced oil recovery, a.k.a. 'EOR'). Enables configurable assumptions about methane leakage, global warming potential of methane (GWP), grid mix, and EOR vs. geologic storage splits.

## Package structure

```
ccs_ghg/
├── __init__.py
├── config.py                        # Top-level config: CCSProjectConfig, CCSConstants,
│                                    #   EmissionFactors, IndustryInput, IndustryConfig
├── constants.py                     # Physical constants (molecular weights, unit conversions,
│                                    #   time constants) — not configurable assumptions
├── ccs_emissions/                   # Module 1: per-(source, destination) CCS emissions factors
│   ├── __init__.py
│   ├── config.py                    # Re-exports CCSConstants and CCSProjectConfig
│   ├── capture.py                   # Emissions from CO2 capture for various industries
│   ├── transport.py                 # Emissions from pipeline transport of CO2
│   ├── storage.py                   # Emissions from injection (EOR and geologic storage)
│   ├── energy_emissions.py          # Grid, gas turbine, and on-site gas emission functions
│   ├── compute.py                   # Orchestrator: assembles raw eor/gs emissions factors
│   └── cli.py                       # Click CLI
├── industrial_emissions/            # Module 2: per-industry CO2/CO2e emissions factors
│   ├── __init__.py
│   ├── config.py                    # ConversionConstants, IndustrialEmissionsConfig;
│   │                                #   re-exports IndustryInput from top-level config
│   ├── compute.py                   # tCO2/tCO2e computation and adjusted capture fractions
│   └── cli.py                       # Click CLI
├── emissions_scenarios/             # Module 3: project-level scenarios (weighting, filtering, 45Q)
│   ├── __init__.py
│   ├── config.py                    # ScenarioConfig, DestinationMix, MethaneScenario
│   ├── scenarios.py                 # Portfolio-level GHG emissions modeling
│   └── cli.py                       # Click CLI
├── config/
│   ├── config_infrastructure.yml    # National + per-state grid/methane constants, EPA
│   │                                #   emission factors, transport & storage energy intensities
│   ├── config_industries.yml        # Per-industry emission profiles and capture cases
│   ├── config_scenarios.yml         # Scenarios config (paths, weighted_scenarios,
│   │                                #   methane_scenarios, 45Q, oil revenue params)
│   ├── config_infrastructure_test_case.yml   # Test-case fixture (mirrors main structure)
│   ├── config_industries_test_case.yml       # Test-case fixture
│   ├── config_scenarios_test_case.yml        # Test-case fixture
│   ├── config_industries_hand_test.yml       # Fixture for hand-regression tests
│   ├── config_scenarios_texas_only.yml       # Scenarios config scoped to Texas projects
│   ├── config_industry_emissions.yml         # Optional pre-computed industry emissions
│   ├── config_plotting.yml                   # Plotting styles/colors
│   └── ccs_projects_test_case.csv            # Test-case project fixture
├── README.md
└── tests/
    ├── test_constants.py             # Physical constants
    ├── test_factors.py               # Grid, gas turbine, on-site gas per-MWh/MMBTU factors
    ├── test_industrial_emissions.py  # Per-industry tCO2/tCO2e math and capture fractions
    ├── test_hand_regression.py       # Hand-worked end-to-end regression suite
    ├── test_pipeline.py              # Precomputed vs. full-pipeline integration
    └── test_scenarios.py             # Scenario weighting, nameplate cases, 45Q, PV
```

## Requirements
- Python 3.9+
- numpy, pandas, click
- A `utils.io` module providing `yaml_to_dict(path) -> dict`

## Data needed
- Emissions information for implementations of CCS according to industry
- Emissions information for non-CCS industrial products
- Facility details (product volume produced annually)
- Assumptions (see configuration)

## Configuration

All configuration is driven from a single YAML file (`config/config_scenarios.yml`) that contains paths to the infrastructure, industries, and CH4 assumptions configs as well as scenario definitions. The three sub-configs are described below.

### Infrastructure config (`config/config_infrastructure.yml`)
All non-industry assumptions in one file, organized into five top-level blocks:

- **`constants`** — national defaults for methane assumptions, grid intensity fallbacks, oil/EOR parameters, and the derived pipeline-gas coefficient `tch4_per_mmbtu_pipeline_gas` (κ). Includes `gwp_ch4`, `methane_leak_rate` (US-average fallback; Alvarez et al. 2018), `grid_tco2_per_mwh`, `grid_tch4_per_mwh`, `grid_ng_heat_rate_mmbtu_per_mwh` (state NG fleet avg heat rate; used by `power_ng_retrofit`), `oil_tco2_per_bbl`, `recovery_factor_bbl_per_tco2`, `eor_gas_turbine_efficiency`, `tch4_per_mmbtu_pipeline_gas`. Does *not* include physical constants like molecular weights — those live in `constants.py`.
- **`emission_factors`** — EPA combustion emission factors: `natural_gas_tco2_per_mmbtu` (national) and `coal_tco2_per_mmbtu` (national fallback; states override with basin-appropriate values). Used to derive power-plant combustion CO2 from heat rates.
- **`states`** — per-state overrides for values that genuinely vary by state (grid CO2 intensity, grid CH4 intensity, coal emission factor, methane leak rate). Keyed by the value in the projects CSV `project_state` column. See "State-specific parameters" below.
- **`transport`** — transport energy intensity at near/average/far distances (MWh/tCO2).
- **`storage`** — injection energy intensity for EOR (open-cycle gas turbine) and geologic storage (grid-powered compressor) (MWh/tCO2).

The file contains an inline derivation of `tch4_per_mmbtu_pipeline_gas = 0.01827` (κ) from EPA/GPSA HHV data, IUPAC atomic weights, and ideal-gas density at natural-gas standard conditions.

### State-specific parameters

Five constants vary by U.S. state under the `states:` block in `config_infrastructure.yml`:

- **`grid_tco2_per_mwh`** — state-average grid CO2 intensity (EPA eGRID2023, state summary tables)
- **`grid_tch4_per_mwh`** — mass of methane burned in state NG plants per MWh of *total* grid electricity generated in the state (derived from EPA eGRID2023 `UNIT23.HTIAN` aggregated across NG units divided by `STNGENAN`, converted via κ)
- **`grid_ng_heat_rate_mmbtu_per_mwh`** — state NG fleet average heat rate (MMBtu of gas burned per MWh of gas-plant electricity generated). Used only by the `power_ng_retrofit` industry variant (see below); other power_ng variants use hardcoded new-build heat rates.
- **`coal_tco2_per_mmbtu`** — coal EF appropriate to the state's coal fleet (e.g. lignite for TX, bituminous/subbituminous for LA)
- **`methane_leak_rate`** — currently uses Alvarez et al. 2018 national average across all states as a conservative baseline; per-state overrides supported (e.g., for Permian/Haynesville basin-specific rates in sensitivity analyses)

Every project in the projects CSV must have a `project_state` value that matches a key in the `states:` block, or the run raises `KeyError` with the list of defined vs. missing states.

Baseline shipped values include `TX` and `LA` blocks. To add another state, add a block under `states:` with the state-specific keys; any field not listed inherits from the top-level `constants:` / `emission_factors:` defaults.

### Industries config (`config/config_industries.yml`)
Per-industry data, grouped by `simple_product_name`:

```yaml
industries:
  hydrogen:
    emission_profiles:
      - industrial_product: hydrogen
        unit: "ton H2"
        process_tco2_per_unit_product: 4.62
        # ... emission profile fields ...
    capture_cases:
      - case_name: refinery_hydrogen_90
        co2_stream_capture_rate: 0.90
        # ... capture case fields ...
```

Each industry has:
- **`emission_profiles`** — one or more emission variants. Fields include process/combustion CO2, CH4 throughput, `ccs_applied_to` (which emission stream is captured), `is_methane_based` (bool; see below), and optional heat rate + fuel type for power plants. Capture fractions are derived at config load time from the primary capture case's `co2_stream_capture_rate`.
- **`capture_cases`** — one or more CCS facility configurations (energy requirements, capture rates, distances). Only `primary_case: 1` entries are used for emissions factor computation.

**Three NG-power variants** are defined:
- `power_ng_retrofit` — heat rate is state-specific (`heat_rate_btu_per_kwh: null` in the YAML triggers a runtime lookup of `grid_ng_heat_rate_mmbtu_per_mwh × 1000`). For retrofitted existing plants.
- `power_ng_new_build` — hardcoded 6,363 Btu/kWh (NETL Fossil Energy Baseline, NGCC F class). For new-build utility plants.
- `power_ng_data_center` — same hardcoded heat rate as `power_ng_new_build`, separately labeled so data-center-serving generation can be attributed distinctly in outputs and plots.

**`is_methane_based` flag** — each industry entry may declare `is_methane_based: true` if natural gas / methane is a material feedstock OR primary process-heat source. The flag propagates through the pipeline to `industrial_emissions.csv`, `ccs_emissions.csv`, `projects.csv`, and `scenarios.csv`, enabling downstream filtering and faceting (e.g., "methane-based vs. non-methane-based industries" breakdown in plots). Currently `true` for: hydrogen, ammonia, iron_steel (DRI-with-NG), methanol, lng, ngp, petrochemicals, all three `power_ng_*` variants, and `dac_with_ngcc`. `false` for: ethanol, refined_fuels, power_coal, dac (electricity-only).

### Scenarios config (`config/config_scenarios.yml`)
The main config file, referenced by all three CLIs via `--config`. Contains:

- **`infrastructure_config_path`** — path to the infrastructure config YAML
- **`industries_config_path`** — path to the industries config YAML
- **`projects_info_path`** — path to facility/project CSV. The CSV **must** include a `project_state` column (every value must match a key in the `states:` block of the infrastructure config) and a `product` column (mapping to a `simple_product_name` in the industries config). An optional `include` column (int, 0/1) filters rows at load time — only `include == 1` rows are processed; the rest are dropped with an INFO log. Rows with missing `project_start` are dropped with a WARNING. The CSV is read with `encoding="latin1"` so Excel-exported files with smart quotes / em-dashes work without pre-cleaning.
- **`emissions_factors_path`** — path to pre-computed emissions factors CSV (output of factors module)
- **`emissions_for_industries_path`** — path to industry emissions CSV
- **`weighted_scenarios`** — list of destination-mix scenarios (e.g. current_mix, all_eor, all_gs)
- **`which_industries`** — list of industries to include in the analysis
- Column mappings and financial parameters (45Q credit, discount rate, oil profit per barrel, etc.)
- **`use_precomputed`** — `true` (default): read pre-computed CSVs. `false`: compute CCS factors and industrial emissions on the fly.
- **`methane_scenarios`** — (optional) list of named methane assumption scenarios to sweep over

#### Methane scenarios

When `methane_scenarios` is defined, each entry specifies a named combination of GWP and methane leak rate:

```yaml
methane_scenarios:
  - name: gwp20_base_leak
    gwp_ch4: 86
    methane_leak_rate: 0.035
  - name: gwp100_low_leak
    gwp_ch4: 29.8
    methane_leak_rate: 0.02
```

All three CLIs support this field. The CCS factors and industrial emissions CLIs loop over methane scenarios and tag output rows with `methane_scenario`, `gwp_ch4`, and `methane_leak_rate` columns. The scenarios CLI in full-pipeline mode produces a **cross-product of states × methane scenarios × destination-mix scenarios**, with output rows carrying `project_state`, `methane_scenario`, `gwp_ch4`, and `methane_leak_rate` tag columns. Without `methane_scenarios`, the CLIs compute once using the baseline values from `constants:` in the infrastructure config.

#### Nameplate scenarios and the source-of-truth flag

Two candidate capture bases exist per project:
- **`implied_capture`** = `unabated_mtco2_per_yr × adjusted_capture_fraction_tco2` — captured tonnes at the industry-standard capture rate.
- **`effective_nameplate`** = operator-supplied `nameplate_captured_mtco2_per_yr` when present in the projects CSV, else `implied_capture` as fallback.

Two dedicated scenarios always appear in outputs, one for each definition:
- **`nameplate_implied`** — captured = `implied_capture` (always the computed value; never uses operator's number)
- **`nameplate`** — captured = `effective_nameplate` (operator when supplied, else implied)

For every project, `nameplate_implied` and `nameplate` are identical when the operator has not published a nameplate; they diverge when the operator has published one.

**The flag `use_operator_nameplate_when_available`** (default `false`) picks the "source of truth" that feeds:
- baseline / current_mix / all_gs / all_eor scenario captured values
- 45Q subsidy basis
- oil barrels and oil revenue

When `false`, all of the above use `implied_capture`. When `true`, they use `effective_nameplate`. The two dedicated nameplate scenarios above are unaffected by the flag.

For DAC projects specifically, all scenarios always use `effective_nameplate` for captured (DAC's unabated is 0 by convention, so `implied_capture` = 0 and the standard formula is meaningless).

#### Financial parameters

The scenarios config also accepts financial parameters used to compute 45Q tax credits and oil revenue:

- **`usd_per_tco2_45q_gs`** — 45Q credit per tonne CO2 captured, geologic storage (default: 85.0)
- **`usd_per_tco2_45q_eor`** — 45Q credit per tonne CO2 captured, EOR (default: 85.0)
- **`recovery_factor_bbl_per_tco2`** — barrels of oil recovered per tonne CO2 injected for EOR (default: 3.0)
- **`profit_per_bbl_usd`** — profit per barrel of oil sold (default: 10.0)
- **`discount_rate`** — discount rate for present-value calculations (default: 0.051)
- **`project_duration`** — project lifetime in years (default: 12)
- **`base_year`** — reference year for discounting (default: 2025)
- **`forty_five_q_earliest_start`** — earliest project start year eligible for 45Q (default: 2023)
- **`use_operator_nameplate_when_available`** — bool (default: `false`). When `true`, baseline/weighted scenarios plus 45Q subsidy plus oil accounting all use `effective_nameplate` (operator value when supplied, else implied). When `false`, all of the above use `implied_capture`. See "Nameplate scenarios and the source-of-truth flag" above.

Oil accounting: both `million_barrels_oil_per_yr` and `million_usd_per_yr_oil_revenue` are computed as `source_capture × eor_fraction × recovery_factor_bbl_per_tco2 × [× profit_per_bbl_usd]`, where `source_capture` is the config-chosen basis (implied vs. effective per the flag above). For baseline, `eor_fraction` is 0 or 1 per project based on stated CO2 destination; for weighted scenarios it is the scenario-level EOR fraction.

## Physical constants vs. configurable assumptions

Physical constants (molecular weights, unit conversions, time constants) live in `constants.py` as module-level Python constants. These are immutable facts that never change across scenarios.

Configurable assumptions (grid CO2 intensity, gas turbine efficiency, methane leak rate, etc.) live in the YAML config files and are loaded into the `CCSConstants` dataclass. National defaults live under `constants:` and `emission_factors:` in the infrastructure config; five of them (grid CO2 intensity, grid CH4 intensity, grid NG fleet heat rate, coal EF, methane leak rate) can be overridden per state via the `states:` block. At runtime, `CCSProjectConfig.constants_for_state(state)` returns the merged (base + state-override) `CCSConstants` for a given state.

## Emissions scope

The model covers Scope 1, Scope 2, and selected Scope 3 emissions. Results are reported in both tCO2 (CO2 only) and tCO2e (CO2 + CH4, converted via GWP). The difference between tCO2 and tCO2e columns reflects the methane (Scope 3 upstream) contribution.

### What is included

| Scope | Category | Where in the model |
|-------|----------|--------------------|
| **Scope 1** | Process CO2 from industrial reactions (e.g., SMR reforming, fermentation, calcination) | `process_tco2_per_unit_product` in `config_industries.yml` |
| **Scope 1** | On-site combustion CO2 for industrial energy | `combustion_tco2_per_unit_product` in `config_industries.yml` (from published LCA sources that include the full energy budget of each industrial process) |
| **Scope 1** | On-site natural gas combustion for CO2 capture | `capture_gas_combustion_tco2` in `ccs_emissions/capture.py` |
| **Scope 1** | Escaped CO2 not captured by CCS | `capture_escaped_tco2` in `ccs_emissions/capture.py` |
| **Scope 2** | Grid electricity for CO2 capture | `capture_electricity_tco2` in `ccs_emissions/capture.py` |
| **Scope 2** | Grid electricity for CO2 pipeline transport | `tco2_emitted_per_tco2_transported` in `ccs_emissions/transport.py` |
| **Scope 2** | Grid electricity for geologic storage injection | `tco2_emitted_per_tco2_stored` (GS) in `ccs_emissions/storage.py` |
| **Scope 3 upstream** | Fugitive CH4 from the natural gas supply chain | `methane_leak_rate × gwp_ch4 × tch4_throughput` applied throughout; appears in all tCO2e columns |
| **Scope 3 downstream** | Combustion of oil produced via EOR | `tco2_emitted_from_ccs_product_per_tco2_ccs` in `ccs_emissions/storage.py` (= `oil_tco2_per_bbl × recovery_factor_bbl_per_tco2`) |

EOR injection is powered by an on-site open-cycle gas turbine (Scope 1), not grid electricity, so `tco2_emitted_per_tco2_stored` for EOR uses `compute_gas_turbine_emissions_per_mwh` rather than grid emission factors.

### What is omitted

- **Scope 1**: Fugitive CO2 from CCS pipelines and injection wells (assumed zero leakage from CCS infrastructure)
- **Scope 3 upstream**: Upstream emissions from non-gas inputs — coal mining and transport (power_coal), limestone extraction (cement), crude oil extraction (refined_fuels, petrochemicals). Only natural gas supply-chain CH4 leaks are modeled.
- **Scope 3 downstream**: Combustion of the industrial product itself (e.g., refined fuels burned by consumers, LNG burned at destination, hydrogen consumed). Only EOR-produced oil combustion is tracked.
- **Scope 3 other**: Embodied emissions in CCS infrastructure (pipeline steel, well construction, compressor manufacturing); transport of industrial products to end users.

## Usage

### Command line

All three CLIs take a single `--config` flag pointing to the scenarios config YAML:

```bash
# Compute per-industry CO2/CO2e emissions factors
python -m projects.ccs_ghg.industrial_emissions.cli \
    --config config/config_scenarios.yml \
    --output industrial_emissions.csv

# Compute CCS emissions factors
python -m projects.ccs_ghg.ccs_emissions.cli \
    --config config/config_scenarios.yml \
    --output emissions_factors.csv

# Compute project-level scenarios (precomputed mode)
python -m projects.ccs_ghg.emissions_scenarios.cli \
    --config config/config_scenarios.yml \
    --output-projects projects.csv \
    --output-scenarios scenarios.csv

# Full-pipeline mode (set use_precomputed: false in config_scenarios.yml)
# Computes factors + industrial emissions for each methane scenario,
# then applies destination-mix weighting
python -m projects.ccs_ghg.emissions_scenarios.cli \
    --config config/config_scenarios.yml \
    --output-projects projects.csv \
    --output-scenarios scenarios.csv

# Export publication-ready tables (industrial emissions, CCS+GS, CCS+EOR)
python -m projects.ccs_ghg.export_tables \
    --config config/config_scenarios.yml \
    --output-industrial table_a_industrial_emissions.csv \
    --output-ccs-gs table_b_ccs_geologic_storage.csv \
    --output-ccs-eor table_c_ccs_eor.csv
```

The scenarios CLI produces two output files:
- **Projects file** (`--output-projects`): project metadata (name, uid, `project_state`, owner, status, lat/lon, carbon fate, dates, product info, co2_source, `is_methane_based`, `nameplate_captured_mtco2_per_yr`).
- **Scenarios file** (`--output-scenarios`): per-(project, scenario, methane_scenario, destination) results, keyed by uid. Includes net emissions (CO2 and CO2e), 45Q credit (annual and PV), oil production, oil revenue (annual, nominal total, PV), and passenger columns `project_state`, `is_methane_based`, `operator_reported_nameplate_captured_mtco2_per_yr`, `methane_scenario`, `gwp_ch4`, `methane_leak_rate`.

As side effects (paths controlled by `ccs_emissions_factors_path` and `emissions_for_industries_path` in the scenarios YAML), the pipeline also writes:
- **CCS emissions factors** — per-(co2_source, co2_destination, distance, state, methane_scenario) lifecycle factors, tagged with `is_methane_based`.
- **Industrial emissions** — per-(industry, state, methane_scenario) per-unit emissions and adjusted capture fractions, tagged with `is_methane_based`.

The export tables CLI produces three CSV files for publication:
- **Table A** (`--output-industrial`): industrial process emissions per unit product — process and combustion CO2 and CH4, capture rate, heat rate and fuel type (for power plants)
- **Table B** (`--output-ccs-gs`): per-industry CCS emissions for the geologic storage pathway, broken into capture sub-components (grid electricity, on-site gas combustion, escaped CO2), pipeline transport, and GS injection — in both tCO2 and tCO2e per tCO2 sequestered
- **Table C** (`--output-ccs-eor`): per-industry CCS emissions for the EOR pathway — same structure as Table B, plus oil combustion emissions from EOR-produced oil

### As a library

```python
from projects.ccs_ghg.ccs_emissions import compute_ccs_emissions_factors
from projects.ccs_ghg.config import CCSProjectConfig

config = CCSProjectConfig.from_yaml(
    "path/to/config_infrastructure.yml",
    "path/to/config_industries.yml",
)
# Optionally scope to one state's constants:
tx_constants = config.constants_for_state("TX")
ef_df = compute_ccs_emissions_factors(config)
```

For industrial emissions:

```python
from projects.ccs_ghg.industrial_emissions import compute_industrial_emissions
from projects.ccs_ghg.industrial_emissions.config import IndustrialEmissionsConfig
from projects.ccs_ghg.config import CCSProjectConfig

config = CCSProjectConfig.from_yaml(
    "path/to/config_infrastructure.yml",
    "path/to/config_industries.yml",
)
ie_config = IndustrialEmissionsConfig.from_project_config(config)
ie_df = compute_industrial_emissions(ie_config)
```

For scenario modelling (precomputed mode):

```python
import pandas as pd
from projects.ccs_ghg.emissions_scenarios import compute_scenarios
from projects.ccs_ghg.emissions_scenarios.config import ScenarioConfig

cfg = ScenarioConfig.from_yaml("path/to/config_scenarios.yml")
projects_df = pd.read_csv(cfg.projects_info_path)
ef_df = pd.read_csv(cfg.emissions_factors_path)

results = compute_scenarios(
    projects_df=projects_df,
    emissions_factors_df=ef_df,
    scenario_config=cfg,
)
```

For the full pipeline (computing factors and industrial emissions on the fly, sweeping methane scenarios):

```python
from projects.ccs_ghg.emissions_scenarios import compute_scenarios_full_pipeline
from projects.ccs_ghg.emissions_scenarios.config import ScenarioConfig

cfg = ScenarioConfig.from_yaml("path/to/config_scenarios.yml")
# cfg.use_precomputed must be False; methane_scenarios may be populated.
# The projects CSV at cfg.projects_info_path must have a project_state column;
# every state value must match a key in the states: block of the infrastructure config.
results = compute_scenarios_full_pipeline(cfg)
# results carries project_state, methane_scenario, gwp_ch4, methane_leak_rate columns
```

