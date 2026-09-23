"""Physical and unit-conversion constants for CCS GHG computations.

These values are mathematical or physical facts that never change.
Configurable assumptions (grid mix, emission factors, policy parameters)
live in the projects' yml configuration files.
"""

# Molecular weights
MOLECULAR_WEIGHT_CO2_G_PER_MOL: float = 44.0095
MOLECULAR_WEIGHT_CH4_G_PER_MOL: float = 16.04246

# Energy unit conversions
GJ_PER_MMBTU: float = 1.055056
MJ_PER_GJ: int = 1000
MJ_PER_MMBTU: float = GJ_PER_MMBTU * MJ_PER_GJ  # 1055.056
MJ_PER_MWH: float = 3600.0
BTU_PER_MMBTU: int = 1_000_000
KWH_PER_MWH: int = 1_000

# Time
DAYS_PER_YEAR: int = 365
HOURS_PER_DAY: int = 24

# Scale factors (not currently used in compute code, but useful for conversions)
TCO2_PER_MTCO2: int = 1_000_000
MWH_PER_TWH: int = 1_000_000
BBL_IN_KBOE: int = 1_000
