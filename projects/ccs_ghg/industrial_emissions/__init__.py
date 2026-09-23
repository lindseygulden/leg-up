"""Industrial emissions computation sub-package."""

from projects.ccs_ghg.industrial_emissions.compute import (
    compute_industrial_emissions,
    compute_industry_row,
    resolve_combustion_co2_tco2,
)
from projects.ccs_ghg.industrial_emissions.config import (
    IndustrialEmissionsConfig,
    IndustryInput,
)

__all__ = [
    "compute_industrial_emissions",
    "compute_industry_row",
    "resolve_combustion_co2_tco2",
    "IndustrialEmissionsConfig",
    "IndustryInput",
]
