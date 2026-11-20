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


# ============================================================================
# Derivation of κ (tCH4 per MMBtu of pipeline natural gas)
#
# Reproduces methods §1.1.2 "Methane mass per MMBtu of natural gas (κ)".
# Pipeline gas is treated as a four-component ideal-gas mixture of CH4,
# C2H6, CO2, and N2. The two inert diluents (2% CO2 + 1% N2) represent
# typical U.S. pipeline-gas tariff limits; trace heavier hydrocarbons are
# still neglected (lumped into the ethane term via the HHV-matching solve).
# ============================================================================

# --- Physical inputs ---------------------------------------------------------
HHV_PIPELINE_GAS_MMBTU_PER_SCF: float = 0.001028
"""Volumetric higher heating value of U.S. pipeline natural gas at standard
conditions (60°F, 14.696 psia) — the same number regulators use for GHGRP
reporting (EPA 40 CFR 98 Table C-1)."""

HHV_CH4_BTU_PER_SCF: float = 1014.0
"""Pure-methane HHV at 60°F and 14.73 psia (natural-gas standard conditions),
from Armstrong & Jobe (1982), NBSIR 82-2401, Table II (also summarized in
GPSA Engineering Data Book, Section 23 reference tables). Burning one scf
of pure methane releases about 1014 Btu."""

HHV_C2H6_BTU_PER_SCF: float = 1787.92
"""Pure-ethane HHV at 60°F and 14.73 psia, same NBSIR source. About 1.76×
methane per unit volume because each ethane molecule packs more chemical
energy."""

MOLAR_MASS_CH4_G_PER_MOL: float = 16.043
"""Methane molecular weight (IUPAC atomic weights, rounded to 3 decimals).
The higher-precision `MOLECULAR_WEIGHT_CH4_G_PER_MOL = 16.04246` above is
used elsewhere in the codebase; the two agree to 0.003%, so downstream
numbers don't move meaningfully either way."""

MOLAR_MASS_C2H6_G_PER_MOL: float = 30.069
"""Ethane molecular weight, same IUPAC source and precision."""

MOLAR_MASS_CO2_G_PER_MOL: float = 44.010
"""Carbon dioxide molecular weight (IUPAC, rounded to 3 decimals). Inert
contaminant in pipeline gas; contributes to density but zero HHV."""

MOLAR_MASS_N2_G_PER_MOL: float = 28.014
"""Nitrogen molecular weight (IUPAC, rounded to 3 decimals). Inert
contaminant in pipeline gas; contributes to density but zero HHV."""

MOLE_FRACTION_CO2_PIPELINE_GAS: float = 0.02
"""Assumed CO2 mole fraction in pipeline gas (2%). Representative of typical
U.S. pipeline tariff limits (gas quality specifications generally cap CO2
at 2–4% to avoid acid corrosion). Enters as inert diluent (zero HHV)."""

MOLE_FRACTION_N2_PIPELINE_GAS: float = 0.01
"""Assumed N2 mole fraction in pipeline gas (1%). Represents trace inerts
beyond CO2 (nitrogen plus any argon, oxygen); together with CO2 totals 3%
non-combustible content, leaving 97% hydrocarbon."""

IDEAL_GAS_MOLAR_VOLUME_SCF_PER_LBMOL: float = 379.3
"""Volume one pound-mole of an ideal gas occupies at natural-gas standard
conditions (60°F). Shows up in every scf ↔ mass conversion for gas."""

KG_PER_LB: float = 0.4535924
"""1 lb = 0.4535924 kg (NIST SP 811). Used to convert the gas density we
compute from lb/scf into kg/scf."""

# --- Derived: pure-component HHVs in MMBtu basis for mole-fraction math ------
HHV_CH4_MMBTU_PER_SCF: float = HHV_CH4_BTU_PER_SCF / BTU_PER_MMBTU
HHV_C2H6_MMBTU_PER_SCF: float = HHV_C2H6_BTU_PER_SCF / BTU_PER_MMBTU

# --- Derived: mole fractions (paper Eq. eq:mole_fraction) --------------------
# Volumetric HHV mixes linearly with mole fraction. With inert fraction
# f_inert = x_CO2 + x_N2 fixed, the hydrocarbon mole fractions sum to
# (1 − f_inert); solving the HHV constraint for the CH4 share:
#     H_gas = x_CH4·H_CH4 + x_C2H6·H_C2H6 + 0·f_inert
#     x_CH4 + x_C2H6 = (1 − f_inert)
# ⇒   x_CH4 = [H_C2H6·(1 − f_inert) − H_gas] / (H_C2H6 − H_CH4)
_INERT_MOLE_FRACTION: float = (
    MOLE_FRACTION_CO2_PIPELINE_GAS + MOLE_FRACTION_N2_PIPELINE_GAS
)
MOLE_FRACTION_CH4_PIPELINE_GAS: float = (
    HHV_C2H6_MMBTU_PER_SCF * (1.0 - _INERT_MOLE_FRACTION)
    - HHV_PIPELINE_GAS_MMBTU_PER_SCF
) / (HHV_C2H6_MMBTU_PER_SCF - HHV_CH4_MMBTU_PER_SCF)
"""What fraction of the molecules in pipeline gas are methane — about
91% by count. Back-solved from the measured pipeline-gas HHV with a
four-component ideal-gas mixture (CH4, C2H6, CO2, N2); the CO2 and N2
mole fractions are fixed at 2% and 1% respectively, and ethane picks up
whatever share of the remaining 97% is needed to match H_gas = 1028 Btu/scf.
Trace propane/butane are still folded into the ethane term."""

MOLE_FRACTION_C2H6_PIPELINE_GAS: float = (
    1.0 - _INERT_MOLE_FRACTION - MOLE_FRACTION_CH4_PIPELINE_GAS
)
"""Ethane mole fraction, derived as (1 − f_inert) − x_CH4. About 5.7% —
much higher than the ~1.8% a binary CH4/C2H6 solve would imply, because
with inerts present ethane must now carry more of the HHV load."""

# --- Derived: mixture molar mass (paper Eq. eq:molar_mass) -------------------
MOLAR_MASS_PIPELINE_GAS_G_PER_MOL: float = (
    MOLE_FRACTION_CH4_PIPELINE_GAS * MOLAR_MASS_CH4_G_PER_MOL
    + MOLE_FRACTION_C2H6_PIPELINE_GAS * MOLAR_MASS_C2H6_G_PER_MOL
    + MOLE_FRACTION_CO2_PIPELINE_GAS * MOLAR_MASS_CO2_G_PER_MOL
    + MOLE_FRACTION_N2_PIPELINE_GAS * MOLAR_MASS_N2_G_PER_MOL
)
"""Average molar mass of pipeline gas, ≈ 17.5 g/mol — a mole-fraction-
weighted mix of CH4 (16.043), C2H6 (30.069), CO2 (44.010), and N2 (28.014).
Heavier than a binary CH4/C2H6 mix (16.3 g/mol) because CO2 is a dense
inert."""

# --- Derived: mass fraction of CH4 (paper Eq. eq:mass_fraction) --------------
MASS_FRACTION_CH4_PIPELINE_GAS: float = (
    MOLE_FRACTION_CH4_PIPELINE_GAS
    * MOLAR_MASS_CH4_G_PER_MOL
    / MOLAR_MASS_PIPELINE_GAS_G_PER_MOL
)
"""What fraction of pipeline gas is methane by mass — about 83.5%. Lower
than the ~91% mole fraction because ethane and CO2 molecules are heavier,
so the same mole fraction converts to a smaller mass fraction."""

# --- Derived: pipeline-gas density (paper Eq. eq:rho) ------------------------
PIPELINE_GAS_DENSITY_KG_PER_SCF: float = (
    MOLAR_MASS_PIPELINE_GAS_G_PER_MOL / IDEAL_GAS_MOLAR_VOLUME_SCF_PER_LBMOL * KG_PER_LB
)
"""Density of pipeline gas at standard conditions — ≈ 21 g per standard
cubic foot. Just the mixture molar mass divided by the ideal-gas molar
volume, converted to kg. The g/mol ↔ lb/lb-mol identity is used silently
in the numerator (since they're numerically equal)."""

# --- Derived: mass-based HHV (paper Eq. eq:mass_hhv) -------------------------
MASS_HHV_PIPELINE_GAS_MMBTU_PER_T: float = (
    HHV_PIPELINE_GAS_MMBTU_PER_SCF / PIPELINE_GAS_DENSITY_KG_PER_SCF * 1000.0
)
"""Heating value of pipeline gas per tonne of gas — about 49 MMBtu/t.
Just the volumetric HHV divided by the density (with kg → t bookkeeping).
Lower than the binary-mix value (~52.75 MMBtu/t) because inert content
raises the mass denominator without contributing any HHV."""

# --- Final: κ (paper Eq. eq:kappa) -------------------------------------------
KAPPA_TCH4_PER_MMBTU: float = (
    MASS_FRACTION_CH4_PIPELINE_GAS / MASS_HHV_PIPELINE_GAS_MMBTU_PER_T
)
"""How much methane (by mass) is carried in each MMBtu of pipeline gas —
about 0.0170 tCH4/MMBtu. Used everywhere the model needs to turn a gas
heat input into the mass of methane that flowed through the supply chain,
so we can attribute fugitive methane emissions to that use of gas. The
paper reports this value in §1.1.2, Eq. (eq:kappa); this expression
reproduces it. Serves as the default for
`CCSConstants.tch4_per_mmbtu_pipeline_gas` when the YAML omits the field."""
