"""Typed configuration for CCS scenario computation."""

# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional

from ccs_ghg._vendored.io import yaml_to_dict


@dataclass
class DestinationMix:
    """A named scenario with distinct eor / geologic-storage weighting."""

    name: str
    destination_pct: Dict[str, float]  # e.g. {"eor": 80, "gs": 20}
    plotting_name: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "DestinationMix":
        """Import DestinationMix dataclass directly from a dictionary."""
        return cls(
            name=d["name"],
            destination_pct=d["destination_pct"],
            plotting_name=d.get("plotting_name"),
        )


@dataclass
class MethaneScenario:
    """A named combination of GWP and methane supply-chain leak-rate assumptions."""

    name: str
    gwp_ch4: float
    methane_leak_rate: float
    leak_plotting_name: Optional[str] = None
    gwp_plotting_name: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "MethaneScenario":
        """add a new methane scenario from a dictionary"""
        return cls(
            name=d["name"],
            gwp_ch4=d["gwp_ch4"],
            methane_leak_rate=d["methane_leak_rate"],
            leak_plotting_name=d.get("leak_plotting_name"),
            gwp_plotting_name=d.get("gwp_plotting_name"),
        )


@dataclass
class ScenarioConfig:
    """Configuration information for CCS scenarios."""

    # --- Data paths ---
    emissions_for_industries_path: str
    projects_info_path: str
    ccs_emissions_factors_path: str
    infrastructure_config_path: Optional[str] = None
    industries_config_path: Optional[str] = None

    # --- Column-name mappings (match CSV schemas) ---
    industry_emissions_cols: List[str] = field(default_factory=list)
    industry_emissions_which_flavor: List[str] = field(default_factory=list)
    facility_keep_cols: List[str] = field(default_factory=list)
    projects_merge_on: str = ""
    ccs_emissions_merge_on: str = ""
    nameplate_ccs_col: str = "nameplate_captured_mtco2_per_yr"
    project_start_col: str = "project_start"
    project_unabated_mtco2e_col: str = "unabated_mtco2e_per_yr"
    product_col_name: str = "simple_product_name"
    industrial_product_with_detail_col_name: str = "industrial_product"
    adjusted_capt_frac_col_name: str = "adjusted_capture_fraction_tco2e"

    # --- Scenario definitions ---
    weighted_scenarios: List[DestinationMix] = field(default_factory=list)
    methane_scenarios: List[MethaneScenario] = field(default_factory=list)
    include_nameplate_case: bool = False
    use_precomputed: bool = True
    # If True: baseline / current_mix / all_gs / all_eor scenarios use the
    # "effective nameplate" (operator-supplied when present, else the implied
    # capture value). 45Q subsidy and oil accounting (both barrels and revenue)
    # inherit the same source. If False (default): all of the above use
    # `implied_capture` (unabated × adjusted_capture_fraction). The dedicated
    # `nameplate_implied` and `nameplate` scenarios always show their own
    # semantics regardless of this flag.
    use_operator_nameplate_when_available: bool = False

    # --- Financial / computation parameters ---
    usd_per_tco2_45q_eor: float = 85.0
    usd_per_tco2_45q_gs: float = 85.0
    # IRC §45Q(b)(1)(A)(ii) grants direct-air-capture facilities a higher
    # credit rate than industrial/power CCS, applicable to both EOR and GS
    # destinations. The 2022 Inflation Reduction Act set this to $180/t,
    # retained by the 2025 One Big Beautiful Bill Act.
    usd_per_tco2_45q_dac: float = 180.0
    # LCOE of utility-scale solar (default $60/MWh), used to compute the
    # equivalent solar energy a facility's 45Q stream could buy. The
    # escalation rate applies year-by-year starting from `base_year`; 0.0%
    # (default) means a flat LCOE in nominal $ for every year of the 12-year
    # 45Q window. Set to the inflation rate (~2.5%) to hold the real-dollar
    # LCOE constant; set higher to represent real cost pressure.
    solar_lcoe_usd_per_mwh: float = 60.0
    solar_lcoe_escalation_rate: float = 0.0
    recovery_factor_bbl_per_tco2: float = 3.0
    profit_per_bbl_usd: float = 10.0
    # Nominal discount rate: 10-year U.S. Treasury rate on July 1, 2026.
    discount_rate: float = 0.0448
    # Valuation date for present values, in decimal years (2026.5 = July 1, 2026).
    pv_valuation_year: float = 2026.5
    escalation_year: int = 2027
    inflation_rate: float = 0.025
    project_duration: int = 12
    facility_lifetime_years: int = 30
    base_year: int = 2025
    forty_five_q_earliest_start: int = 2023

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioConfig":
        """Import ScenarioConfig dataclass from dictionary."""
        field_names = {f.name for f in fields(cls)}
        skip = {"weighted_scenarios", "methane_scenarios"}
        kwargs = {k: d[k] for k in field_names if k in d and k not in skip}
        kwargs["weighted_scenarios"] = [
            DestinationMix.from_dict(s) for s in d.get("weighted_scenarios", [])
        ]
        kwargs["methane_scenarios"] = [
            MethaneScenario.from_dict(s) for s in d.get("methane_scenarios", [])
        ]
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "ScenarioConfig":
        """Read config from a YAML file."""
        raw = yaml_to_dict(path)
        return cls.from_dict(raw)
