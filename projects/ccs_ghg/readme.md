# Code and data for "Federally subsidized carbon capture and storage supports methane-based industrial buildout and provides little or no emissions benefit"

This repository contains the code, configuration, and intermediate data
needed to reproduce every figure and table in the paper. The analysis
models the greenhouse-gas (GHG) emissions consequences of the 86
federally subsidized CCS-linked industrial facilities announced for the
U.S. Gulf Coast (Texas + Louisiana) as of mid 2026, under a range of
methane-leakage, CO₂-destination, and 45Q-eligibility assumptions.

## Quick start

```bash
# 1. Clone and enter the directory
git clone <repo-url>
cd ccs_ghg_paper
```

**Install dependencies.** Python 3.9+ is required. Use either tool:

```bash
# Option A — pip (works everywhere, no extra tooling)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Option B — Poetry (if you already use it)
poetry install
```

Both commands install the same runtime and dev dependencies from
`pyproject.toml`. The `-e` flag in pip's case makes the `ccs_ghg`
package editable so you can tweak and re-run without reinstalling.

**Open the driver notebook:**

```bash
jupyter lab notebooks/ccs_ghg.ipynb        # pip install
# or
poetry run jupyter lab notebooks/ccs_ghg.ipynb
```

Run all cells. The notebook reads the two CSVs in `data/`, calls
builders in `helpers/figs_tables.py`, and writes every paper figure
(PNG, 600 dpi) to `figures_for_paper/` plus each table to the clipboard
as a pandas DataFrame.

No per-user path edits are required — see **Paths** below for
optional overrides.

## What's included

```
ccs_ghg_paper/
├── README.md                        # this file
├── LICENSE                          # CC BY-NC 4.0
├── pyproject.toml                   # package metadata + dependencies
├── ccs_ghg/                         # installable Python package
│   ├── __init__.py                  # auto-registers path env vars
│   ├── paths.py                     # DEV_DIR / DATA_DIR — portable paths
│   ├── config.py                    # top-level dataclasses (CCSProjectConfig, etc.)
│   ├── constants.py                 # physical constants (molecular weights, HHVs)
│   ├── plotting.py                  # shared plotting library
│   ├── ccs_emissions/               # CCS factor pipeline (capture + transport + storage)
│   ├── industrial_emissions/        # Per-industry emissions pipeline
│   ├── emissions_scenarios/         # Facility-level scenario engine (45Q, PV, oil)
│   ├── helpers/
│   │   ├── __init__.py
│   │   └── figs_tables.py           # All 24 figure/table builders (standalone)
│   └── _vendored/                   # Vendored subset of leg-up-private/utils
├── config/
│   ├── config_infrastructure.yml           # National + per-state constants
│   ├── config_industries.yml               # Per-industry emission profiles
│   ├── config_scenarios.yml                # Main scenarios config
│   ├── config_infrastructure_eor_disp_0p5.yml   # Table S3 sensitivity
│   └── config_scenarios_eor_disp_0p5.yml        # Table S3 sensitivity
├── data/
│   ├── ccs_scenarios_out.csv                      # Pipeline output for main analysis
│   └── ccs_scenarios_out_eor_disp_0p5.csv         # Pipeline output for Table S3
├── notebooks/
│   └── ccs_ghg.ipynb                # Driver notebook — all paper figures + tables
├── paper/
│   ├── methods_2.tex                # Methods section (paper manuscript)
│   └── table_industry_summary.tex   # Per-industry summary table
└── tests/                           # Regression + unit test suite (optional)
```

## Reproducing paper figures and tables

Open `notebooks/ccs_ghg.ipynb` and run top to bottom. The notebook
sections map one-to-one to the paper:

| Paper artifact | Notebook section |
|---|---|
| Figure 1 (facility map) | ### Figure 1 |
| Figure 3 (state totals, GWP20) | ### Figure 3 |
| Figure 4 (capture reach + cost) | ### Figure 4 |
| Figure 5 (emissions through time, GWP20) | ### Figure 5 |
| Figure 6 (lifetime vs. leak rate) | ### Figure 6 |
| Figure 7 (profit vs. emissions) | ### Figure 7 |
| Figure 8 (lifetime Δ vs. no-CCS) | ### Figure 8 |
| Figures S1–S5 | ### Figure S1 … S5 |
| Tables 1, 4–8 | ### Table 1 … 8 |
| Tables S1–S3 | ### Table S1 … S3 |

Figure 2 in the paper is a hand-drawn schematic and has no code.

Each figure saves to `figures_for_paper/fig{N}.png` or `figs{N}.png`
at 600 dpi (publication-grade PNG). Each table DataFrame is copied to
the clipboard by `.to_clipboard()` for easy paste into a manuscript.

## Paths

Two directories are configurable per machine. Both are exposed as
`pathlib.Path` objects in `ccs_ghg/paths.py`:

- **`DEV_DIR`** — the repo root. Auto-derived from the file's own
  location; never needs to be set by hand.
- **`DATA_DIR`** — external data root (e.g. shapefiles, auxiliary
  CSVs). Defaults to `<DEV_DIR>/data`. Override with:

  ```bash
  export CCS_GHG_DATA_DIR=/path/to/your/data
  ```

Importing anything under `ccs_ghg` registers both as environment
variables (`CCS_GHG_DEV_DIR`, `CCS_GHG_DATA_DIR`), so YAML configs can
reference them as `${CCS_GHG_DEV_DIR}` / `${CCS_GHG_DATA_DIR}` and have
the placeholders expanded at load time. All shipped configs use this
pattern — you should not need to edit any path by hand.

## External data (shapefiles, prices)

A handful of cells in `notebooks/ccs_ghg.ipynb` read bulk geospatial
shapefiles and auxiliary CSVs that are too large (or too licensing-
constrained) to commit to this repository. These feed Figure 1 (facility
map) and a few informational checks; **the paper's headline figures and
tables that come from the scenario pipeline do not require any of them**.

Download them once and point the `CCS_GHG_DATA_DIR` environment variable
at the directory that contains all six paths below:

| Expected path (under `CCS_GHG_DATA_DIR`) | Source | Notes |
|---|---|---|
| `us_geo/high_res_us_states.geojson` | U.S. Census Bureau Cartographic Boundary Files, 2023, 1:500k resolution | <https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip> — convert to GeoJSON with `ogr2ogr`, or use the equivalent Natural Earth 1:10m Admin-1 States file renamed to `high_res_us_states.geojson`. |
| `world_geo/ne_10m_lakes/ne_10m_lakes.shp` | Natural Earth 1:10m Physical — Lakes (+ reservoirs) | <https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-lakes/> |
| `world_geo/ne_10m_lakes_north_america/ne_10m_lakes_north_america.shp` | Natural Earth 1:10m Physical — North America supplement | <https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-lakes/> (companion shapefile) |
| `world_geo/ne_10m_rivers_north_america/ne_10m_rivers_north_america.shp` | Natural Earth 1:10m Physical — North America supplement | <https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-rivers-lake-centerlines/> |
| `oilgas/wti_daily_price_2016_2026.csv` | FRED series `DCOILWTICO` (WTI daily spot) | `curl -o wti_daily_price_2016_2026.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO"` — subset to 2016–2026 if desired. |
| `ccs/global_ccs_institute_operational_facilities_2025.csv` | Global CCS Institute, *Global Status of CCS 2025*, "Operational Facilities" table | Free account at <https://www.globalccsinstitute.com/co2re-tool/> → facility database export → save as CSV. Used for an informational cross-check only. |

After downloading, your `CCS_GHG_DATA_DIR` layout should look like:

```
$CCS_GHG_DATA_DIR/
├── us_geo/
│   └── high_res_us_states.geojson
├── world_geo/
│   ├── ne_10m_lakes/              (full Natural Earth zip contents)
│   ├── ne_10m_lakes_north_america/
│   └── ne_10m_rivers_north_america/
├── oilgas/
│   └── wti_daily_price_2016_2026.csv
└── ccs/
    └── global_ccs_institute_operational_facilities_2025.csv
```

If `CCS_GHG_DATA_DIR` is left unset, the notebook still runs — the
facility-map cell will raise a missing-file error on first execution,
and the WTI/GCCSI cross-check cells will similarly fail, but every other
figure and every table builds normally from the pipeline CSVs in
`data/`.

## Methodology summary

**Scope coverage.** The model covers Scope 1, Scope 2, and selected
Scope 3 emissions. Results are reported in both tCO₂ (CO₂ only) and
tCO₂e (CO₂ + CH₄ via global-warming-potential conversion).

| Scope | Category |
|---|---|
| Scope 1 | Process CO₂ (industrial reactions) |
| Scope 1 | On-site combustion CO₂ for industrial energy |
| Scope 1 | On-site natural gas combustion for CO₂ capture |
| Scope 1 | Escaped CO₂ not captured by CCS |
| Scope 2 | Grid electricity for CO₂ capture |
| Scope 2 | Grid electricity for CO₂ pipeline transport |
| Scope 2 | Grid electricity for geologic-storage injection |
| Scope 3 upstream | Fugitive CH₄ from the natural-gas supply chain |
| Scope 3 downstream | Combustion of EOR-produced oil |

Omitted: fugitive emissions from CCS infrastructure itself, upstream
emissions from non-gas inputs (coal mining, crude extraction),
combustion of the industrial end-product, and embodied emissions of
pipelines/wells.

**Methane scenarios.** Every annual and lifetime emissions figure is
produced for both GWP₂₀ and GWP₁₀₀ using the leak rates from Alvarez
et al. 2018 (`methane_leak_rate = 0.023`). Sensitivities to alternative
leak-rate assumptions (industry-cited, Texas-specific, 3.5%) are
carried as separate `methane_scenario` columns in the output CSVs.

**Destination scenarios.** Six CCS-destination mixes are evaluated:
`baseline` (announced destinations: 17% EOR, 83% GS), `current_mix`
(current U.S. practice: 92% EOR, 8% GS), `all_gs`, `all_eor`,
`switching` (baseline with a pivot to 100% EOR after the 12-year 45Q
window), and `switching_from_current` (same pivot from current practice).
Counterfactuals: `existing_only` (retrofit-eligible facilities only) and
`unabated` (full buildout without CCS).

**EOR oil displacement sensitivity.** The paper's main numbers assume
no displacement of conventional oil by EOR-produced oil (`δ_disp = 0`).
Table S3 reports the sensitivity case `δ_disp = 0.5` per Almutairi &
Pierru 2024.

**Financial accounting.** 45Q credits are computed with the GS/EOR/DAC
schedule in IRC §45Q(b)(1), escalated by inflation from 2027 onward,
discounted to 2025 dollars. Oil revenue uses a constant real
`profit_per_bbl_usd = $10` and `recovery_factor_bbl_per_tco2 = 3.0`.

## Rebuilding the pipeline CSVs from configs

The two CSVs in `data/` are pre-computed outputs of the pipeline. To
regenerate them from the config YAMLs (drop `poetry run ` if you
installed with pip):

```bash
# Main analysis
python -m ccs_ghg.emissions_scenarios.cli \
    --config config/config_scenarios.yml \
    --output-projects data/ccs_projects_out.csv \
    --output-scenarios data/ccs_scenarios_out.csv

# Table S3 sensitivity (EOR displacement = 0.5)
python -m ccs_ghg.emissions_scenarios.cli \
    --config config/config_scenarios_eor_disp_0p5.yml \
    --output-projects data/ccs_projects_out_eor_disp_0p5.csv \
    --output-scenarios data/ccs_scenarios_out_eor_disp_0p5.csv
```

## Configuration details

Every assumption lives in a YAML file under `config/`. The three
sub-configs are:

- **`config_infrastructure.yml`** — national + per-state grid CO₂
  intensities, grid CH₄ intensities, NG fleet heat rates, coal
  emission factors, methane leak rates. Per-state overrides live under
  `states:` (`TX`, `LA` shipped).
- **`config_industries.yml`** — per-industry emission profiles (process
  CO₂, combustion CO₂, CH₄ throughput as feedstock and fuel) and
  capture-case parameters (capture rate, electricity/gas requirements,
  transport distance). Covers ammonia, LNG, hydrogen (SMR), ethanol,
  refining, petrochemicals, iron/steel, natural gas processing,
  methanol, NG power, coal power, and DAC.
- **`config_scenarios.yml`** — the top-level config consumed by all
  CLIs. Points at the two sub-configs above, lists destination
  scenarios, carries 45Q / PV / oil parameters, and (optionally)
  defines methane-scenario sweeps.

See the detailed field-by-field reference in the companion
`methods_2.tex` and the heavily-commented YAML files themselves.

## Running the test suite (optional)

```bash
pytest tests/ -v              # pip install
# or
poetry run pytest tests/ -v   # poetry install
```

The suite covers physical constants, per-industry emission math,
scenario weighting, 45Q escalation and present-value discounting,
equivalent-solar accounting, state-override merging, and hand-worked
regression cases across every industry variant.

## Citation

If you use this code, please cite:

> L. Gulden, J. Buonocore, and C. Harvey, "Federally subsidized carbon
> capture and storage will spur methane-based industrial buildout and
> provide little or no emissions benefit," [journal / preprint], 2026.
> [DOI]

## License

Code and data are released under the Creative Commons
Attribution-NonCommercial 4.0 International License
(CC BY-NC 4.0, <https://creativecommons.org/licenses/by-nc/4.0/>).
You are free to copy, modify, and redistribute the material in any
medium or format for **non-commercial academic use**, provided you give
appropriate credit (see **Citation** above) and indicate if changes
were made. Commercial use is not permitted without a separate written
agreement with the authors.

## Contact

Correspondence: Lindsey Gulden (`gulden@mit.edu`).
