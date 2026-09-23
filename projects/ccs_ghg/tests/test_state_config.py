"""Tests for state-scoped configuration failure modes.

Guards the two contracts of the per-state constants machinery:
  (a) CCSProjectConfig.from_yaml raises ValueError if the infra YAML defines
      no `states:` block (a run must specify at least one state);
  (b) CCSProjectConfig.constants_for_state raises KeyError with an actionable
      message when asked for a state that is not defined.
"""

import textwrap
from pathlib import Path

import pytest

from projects.ccs_ghg.config import CCSProjectConfig

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_INFRASTRUCTURE_CONFIG = str(_CONFIG_DIR / "config_infrastructure_test_case.yml")
_INDUSTRIES_CONFIG = str(_CONFIG_DIR / "config_industries_test_case.yml")


class TestMissingStatesBlock:
    """Loading an infra YAML with no `states:` block must fail loudly."""

    def test_raises_value_error(self, tmp_path):
        infra_no_states = tmp_path / "infra_no_states.yml"
        infra_no_states.write_text(
            textwrap.dedent(
                """
                constants:
                  case_name: no_states_test
                  gwp_ch4: 82.5
                  methane_leak_rate: 0.023
                  grid_tco2_per_mwh: 0.37
                  grid_tch4_per_mwh: 0.08
                  oil_tco2_per_bbl: 0.43
                  recovery_factor_bbl_per_tco2: 3
                  eor_gas_turbine_efficiency: 0.33
                  tch4_per_mmbtu_pipeline_gas: 0.01827
                emission_factors:
                  natural_gas_tco2_per_mmbtu: 0.05306
                  coal_tco2_per_mmbtu: 0.09772
                storage:
                  - co2_destination: eor
                    mwh_per_tco2_stored: 0.25
                  - co2_destination: gs
                    mwh_per_tco2_stored: 0.2847
                transport:
                  - distance: 'near'
                    mwh_per_tco2_transported: 0.09
                """
            ).strip()
        )

        with pytest.raises(ValueError, match="states:"):
            CCSProjectConfig.from_yaml(str(infra_no_states), _INDUSTRIES_CONFIG)


class TestConstantsForUnknownState:
    """constants_for_state('CA') must raise KeyError with actionable message."""

    @pytest.fixture
    def project_config(self):
        return CCSProjectConfig.from_yaml(_INFRASTRUCTURE_CONFIG, _INDUSTRIES_CONFIG)

    def test_raises_key_error_for_unknown_state(self, project_config):
        with pytest.raises(KeyError):
            project_config.constants_for_state("CA")

    def test_error_message_lists_defined_states(self, project_config):
        with pytest.raises(KeyError) as excinfo:
            project_config.constants_for_state("XX")
        # Message should name the missing state and the defined ones.
        msg = str(excinfo.value)
        assert "XX" in msg
        for state in project_config.state_constants:
            assert state in msg

    def test_known_state_returns_constants(self, project_config):
        # Sanity: a known state does not raise, and returns a CCSConstants with
        # the state field populated.
        tx = project_config.constants_for_state("TX")
        assert tx.state == "TX"
