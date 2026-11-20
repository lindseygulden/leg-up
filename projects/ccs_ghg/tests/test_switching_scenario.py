"""Coverage for the synthetic `switching` destination scenario.

The switching scenario represents a facility that follows its baseline stated
destination mix for the first `D = project_duration` years (while 45Q is
earning credit) and then operates CCS at 100% EOR for the remaining
`L − D = facility_lifetime_years − project_duration` years.

These tests cover the three places the implementation touches:

1. The row-synthesis helper `_append_switching_scenario` in
   `projects/ccs_ghg/emissions_scenarios/scenarios.py` (unit tests on a
   hand-built fixture; `TestSwitchingUnit`).
2. The pipeline integration that writes the full
   `ccs_scenarios_out.csv`, including the three new `lifetime_*_oil*`
   columns on every row (`TestSwitchingIntegration`).
3. The downstream `summarize_scenarios` function in `plotting.py`, which
   must consume the new lifetime columns for switching rows
   (`TestSwitchingSummarize`).

Run with: pytest projects/ccs_ghg/tests/test_switching_scenario.py
"""

# pylint: disable=redefined-outer-name
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from ccs_ghg.emissions_scenarios.config import ScenarioConfig
from ccs_ghg.emissions_scenarios.scenarios import (
    _append_switching_scenario,
    nominal_total,
    present_value,
)

# ---------------------------------------------------------------------------
# Shared constants used by the hand-built fixture.
# ---------------------------------------------------------------------------

D = 12  # project_duration (45Q window)
L = 30  # facility_lifetime_years
DISCOUNT_RATE = 0.0448
INFLATION_RATE = 0.025
PV_VALUATION_YEAR = 2026.5
START_YEAR = 2027  # project_start for both fixture facilities

TIME_WEIGHTED_COLS = (
    "eor_fraction",
    "tco2e_emitted_per_tco2_ccs",
    "tco2_emitted_per_tco2_ccs",
    "net_tco2e_emitted_per_tco2_ccs",
    "net_tco2_emitted_per_tco2_ccs",
    "mtco2e_per_yr_emitted_because_of_ccs",
    "mtco2e_per_yr_emissions_including_ccs",
    "mtco2_per_yr_emitted_because_of_ccs",
    "mtco2_per_yr_emissions_including_ccs",
    "million_barrels_oil_per_yr",
    "million_usd_per_yr_oil_revenue",
    "tch4_per_yr_emissions_including_ccs",
)


# ---------------------------------------------------------------------------
# Fixtures for the unit tests on `_append_switching_scenario`
# ---------------------------------------------------------------------------


def _make_row(scenario: str, facility: str, stated: str, overrides: dict) -> dict:
    """Build a single scenarios_out-style row with sensible per-scenario defaults.

    Two facilities are used:
      * Facility A has `stated=eor`, so its baseline row carries non-zero oil.
      * Facility B has `stated=gs`, so its baseline row has zero oil.
    The `all_eor` rows are identical in oil accounting regardless of stated.
    """
    base = {
        "project_name": facility,
        "project_state": "TX",
        "methane_scenario": "edf20",
        "gwp_ch4": 82.5,
        "methane_leak_rate": 0.023,
        "scenario": scenario,
        "stated_co2_destination": stated,
        "assumed_co2_destination": overrides.get("assumed_co2_destination", stated),
        "project_start": START_YEAR,
        "retrofit_newbuild": "new build",
        "is_methane_based": False,
        "adjusted_capture_fraction_tco2": 0.5,
        "adjusted_capture_fraction_tco2e": 0.5,
        "mtco2_per_yr_captured_with_ccs": 2.0,
        "mwh_solar_from_45q": 2.0e7,
        "mtco2e_avoided_from_solar": 5.0,
        "avg_million_usd_per_yr_45q": 170.0,
        "pv_million_usd_credits_45q": 1500.0,
        "unabated_mtco2_per_yr": 4.0,
        "unabated_mtco2e_per_yr": 5.0,
        "unabated_tch4_per_yr": 0.002,
        "lifetime_unabated_mtco2e": L * 5.0,
    }
    base.update(overrides)
    return base


@pytest.fixture
def switching_input_frame():
    """Minimal `scenarios_out`-style frame prior to the switching append."""
    rows = []
    # ---- Facility A: stated = EOR; baseline == all_eor on oil side ----
    for scen, net_e, net_c, bbl, pv_oil, nom_oil, ccs_e in [
        ("baseline", 3.0, 2.5, 3.0, 220.0, 500.0, 1.0),
        ("all_eor", 3.0, 2.5, 3.0, 220.0, 500.0, 1.0),
    ]:
        rows.append(
            _make_row(
                scen,
                "A",
                stated="eor",
                overrides={
                    "eor_fraction": 1.0,
                    "mtco2e_per_yr_emissions_including_ccs": net_e,
                    "mtco2_per_yr_emissions_including_ccs": net_c,
                    "mtco2e_per_yr_emitted_because_of_ccs": ccs_e,
                    "mtco2_per_yr_emitted_because_of_ccs": 0.8,
                    "million_barrels_oil_per_yr": bbl,
                    "million_usd_per_yr_oil_revenue": bbl * 10.0,
                    "nominal_million_usd_oil_revenue": nom_oil,
                    "pv_million_usd_oil_revenue": pv_oil,
                    "tco2e_emitted_per_tco2_ccs": 0.5,
                    "tco2_emitted_per_tco2_ccs": 0.4,
                    "net_tco2e_emitted_per_tco2_ccs": -0.5,
                    "net_tco2_emitted_per_tco2_ccs": -0.6,
                    "tch4_per_yr_emissions_including_ccs": 0.001,
                    "lifetime_mtco2e": D * net_e + (L - D) * 5.0,
                    "lifetime_mtco2_captured_with_ccs": D * 2.0,
                    "lifetime_million_barrels_oil": D * bbl,
                    "lifetime_nominal_million_usd_oil_revenue": nom_oil,
                    "lifetime_pv_million_usd_oil_revenue": pv_oil,
                },
            )
        )
    # ---- Facility B: stated = GS; baseline has zero oil ----
    baseline_B = _make_row(
        "baseline",
        "B",
        stated="gs",
        overrides={
            "eor_fraction": 0.0,
            "mtco2e_per_yr_emissions_including_ccs": 1.5,
            "mtco2_per_yr_emissions_including_ccs": 1.2,
            "mtco2e_per_yr_emitted_because_of_ccs": 0.3,
            "mtco2_per_yr_emitted_because_of_ccs": 0.2,
            "million_barrels_oil_per_yr": 0.0,
            "million_usd_per_yr_oil_revenue": 0.0,
            "nominal_million_usd_oil_revenue": 0.0,
            "pv_million_usd_oil_revenue": 0.0,
            "tco2e_emitted_per_tco2_ccs": 0.1,
            "tco2_emitted_per_tco2_ccs": 0.08,
            "net_tco2e_emitted_per_tco2_ccs": -0.9,
            "net_tco2_emitted_per_tco2_ccs": -0.92,
            "tch4_per_yr_emissions_including_ccs": 0.001,
            "lifetime_mtco2e": D * 1.5 + (L - D) * 5.0,
            "lifetime_mtco2_captured_with_ccs": D * 2.0,
            "lifetime_million_barrels_oil": 0.0,
            "lifetime_nominal_million_usd_oil_revenue": 0.0,
            "lifetime_pv_million_usd_oil_revenue": 0.0,
        },
    )
    all_eor_B = _make_row(
        "all_eor",
        "B",
        stated="gs",
        overrides={
            "eor_fraction": 1.0,
            "mtco2e_per_yr_emissions_including_ccs": 3.5,
            "mtco2_per_yr_emissions_including_ccs": 2.9,
            "mtco2e_per_yr_emitted_because_of_ccs": 1.2,
            "mtco2_per_yr_emitted_because_of_ccs": 0.9,
            "million_barrels_oil_per_yr": 3.0,
            "million_usd_per_yr_oil_revenue": 30.0,
            "nominal_million_usd_oil_revenue": 500.0,
            "pv_million_usd_oil_revenue": 220.0,
            "tco2e_emitted_per_tco2_ccs": 0.5,
            "tco2_emitted_per_tco2_ccs": 0.4,
            "net_tco2e_emitted_per_tco2_ccs": -0.5,
            "net_tco2_emitted_per_tco2_ccs": -0.6,
            "tch4_per_yr_emissions_including_ccs": 0.003,
            "lifetime_mtco2e": D * 3.5 + (L - D) * 5.0,
            "lifetime_mtco2_captured_with_ccs": D * 2.0,
            "lifetime_million_barrels_oil": D * 3.0,
            "lifetime_nominal_million_usd_oil_revenue": 500.0,
            "lifetime_pv_million_usd_oil_revenue": 220.0,
        },
    )
    rows.extend([baseline_B, all_eor_B])
    return pd.DataFrame(rows)


@pytest.fixture
def switching_config():
    """Config with explicit D, L, discount, and inflation parameters."""
    return ScenarioConfig(
        emissions_for_industries_path="dummy",
        projects_info_path="dummy",
        ccs_emissions_factors_path="dummy",
        project_duration=D,
        facility_lifetime_years=L,
        discount_rate=DISCOUNT_RATE,
        inflation_rate=INFLATION_RATE,
        pv_valuation_year=PV_VALUATION_YEAR,
        project_start_col="project_start",
    )


@pytest.fixture
def switching_result(switching_input_frame, switching_config):
    """Frame after appending switching rows."""
    return _append_switching_scenario(switching_input_frame, switching_config)


# ---------------------------------------------------------------------------
# Unit tests #1-15 on `_append_switching_scenario`
# ---------------------------------------------------------------------------


class TestSwitchingUnit:
    # ---- #1 -----------------------------------------------------------
    def test_appends_one_switching_row_per_facility_methane(
        self, switching_input_frame, switching_result
    ):
        key_cols = ["project_name", "project_state", "methane_scenario"]
        expected = switching_input_frame[
            switching_input_frame["scenario"] == "baseline"
        ][key_cols].drop_duplicates()
        new = switching_result[switching_result["scenario"] == "switching"][key_cols]
        assert len(new) == len(expected)
        assert set(map(tuple, new.values)) == set(map(tuple, expected.values))

    # ---- #2 -----------------------------------------------------------
    def test_does_not_mutate_existing_rows(
        self, switching_input_frame, switching_result
    ):
        orig = switching_input_frame.reset_index(drop=True)
        same_cols = [c for c in orig.columns if c in switching_result.columns]
        kept = switching_result[
            switching_result["scenario"].isin(["baseline", "all_eor"])
        ][same_cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            kept.sort_values(["project_name", "scenario"]).reset_index(drop=True),
            orig.sort_values(["project_name", "scenario"])[same_cols].reset_index(
                drop=True
            ),
            check_like=True,
        )

    # ---- #3 -----------------------------------------------------------
    def test_scenario_tag_set(self, switching_result):
        sw = switching_result[switching_result["scenario"] == "switching"]
        assert (sw["scenario"] == "switching").all()
        assert (sw["assumed_co2_destination"] == "switching").all()

    # ---- #4 -----------------------------------------------------------
    @pytest.mark.parametrize(
        "col",
        [
            "project_state",
            "project_start",
            "retrofit_newbuild",
            "methane_scenario",
            "gwp_ch4",
            "methane_leak_rate",
            "unabated_mtco2e_per_yr",
            "unabated_mtco2_per_yr",
            "lifetime_unabated_mtco2e",
        ],
    )
    def test_metadata_copied_from_baseline(self, switching_result, col):
        base = switching_result[switching_result["scenario"] == "baseline"].set_index(
            "project_name"
        )
        sw = switching_result[switching_result["scenario"] == "switching"].set_index(
            "project_name"
        )
        for name in sw.index:
            assert sw.loc[name, col] == base.loc[name, col]

    # ---- #5 -----------------------------------------------------------
    @pytest.mark.parametrize(
        "col",
        [
            "avg_million_usd_per_yr_45q",
            "pv_million_usd_credits_45q",
            "mwh_solar_from_45q",
            "mtco2e_avoided_from_solar",
        ],
    )
    def test_45q_columns_copied_from_baseline(self, switching_result, col):
        """45Q runs only during the 12-year window at the stated destination
        mix, so these credit/solar columns copy from baseline unchanged
        (no tail-year credit)."""
        base = switching_result[switching_result["scenario"] == "baseline"].set_index(
            "project_name"
        )
        sw = switching_result[switching_result["scenario"] == "switching"].set_index(
            "project_name"
        )
        for name in sw.index:
            assert sw.loc[name, col] == pytest.approx(base.loc[name, col], rel=1e-9)

    @pytest.mark.parametrize(
        "col, lifetime_col",
        [
            (
                "nominal_million_usd_oil_revenue",
                "lifetime_nominal_million_usd_oil_revenue",
            ),
            ("pv_million_usd_oil_revenue", "lifetime_pv_million_usd_oil_revenue"),
        ],
    )
    def test_oil_revenue_columns_use_full_lifetime_for_switching(
        self, switching_result, col, lifetime_col
    ):
        """Oil flows for all L years under switching (baseline mix for years
        1..D, 100% EOR for years D+1..L), so the `nominal` and `pv` oil-
        revenue columns carry the full-lifetime total, not baseline's 12-year
        total. They must equal the dedicated `lifetime_*` counterparts."""
        sw = switching_result[switching_result["scenario"] == "switching"]
        pd.testing.assert_series_equal(
            sw[col].reset_index(drop=True),
            sw[lifetime_col].reset_index(drop=True),
            check_names=False,
        )

    # ---- #6 -----------------------------------------------------------
    def test_lifetime_mtco2e_is_exact_two_period_sum(self, switching_result):
        for name in ("A", "B"):
            b = switching_result.query(
                "scenario == 'baseline' and project_name == @name"
            ).iloc[0]
            t = switching_result.query(
                "scenario == 'all_eor' and project_name == @name"
            ).iloc[0]
            s = switching_result.query(
                "scenario == 'switching' and project_name == @name"
            ).iloc[0]
            expected = (
                D * b["mtco2e_per_yr_emissions_including_ccs"]
                + (L - D) * t["mtco2e_per_yr_emissions_including_ccs"]
            )
            assert s["lifetime_mtco2e"] == pytest.approx(expected, rel=1e-9)

    # ---- #7 -----------------------------------------------------------
    def test_lifetime_captured_is_exact_two_period_sum(self, switching_result):
        for name in ("A", "B"):
            b = switching_result.query(
                "scenario == 'baseline' and project_name == @name"
            ).iloc[0]
            t = switching_result.query(
                "scenario == 'all_eor' and project_name == @name"
            ).iloc[0]
            s = switching_result.query(
                "scenario == 'switching' and project_name == @name"
            ).iloc[0]
            expected = (
                D * b["mtco2_per_yr_captured_with_ccs"]
                + (L - D) * t["mtco2_per_yr_captured_with_ccs"]
            )
            assert s["lifetime_mtco2_captured_with_ccs"] == pytest.approx(
                expected, rel=1e-9
            )

    # ---- #8 -----------------------------------------------------------
    def test_lifetime_million_barrels_oil_is_exact_two_period_sum(
        self, switching_result
    ):
        for name in ("A", "B"):
            b = switching_result.query(
                "scenario == 'baseline' and project_name == @name"
            ).iloc[0]
            t = switching_result.query(
                "scenario == 'all_eor' and project_name == @name"
            ).iloc[0]
            s = switching_result.query(
                "scenario == 'switching' and project_name == @name"
            ).iloc[0]
            expected = (
                D * b["million_barrels_oil_per_yr"]
                + (L - D) * t["million_barrels_oil_per_yr"]
            )
            assert s["lifetime_million_barrels_oil"] == pytest.approx(
                expected, rel=1e-9
            )

    # ---- #9 -----------------------------------------------------------
    def test_lifetime_nominal_oil_revenue_is_escalated_two_period_sum(
        self, switching_result, switching_config
    ):
        """Switching's lifetime nominal oil revenue must apply oil-price
        escalation year-by-year across both sub-periods (methods
        Eq.~switching_oil_rev), not just sum base rates.

        Years y_j..y_j+D-1 contribute the baseline D-year escalated nominal
        (already in ``nominal_million_usd_oil_revenue``); years y_j+D..y_j+L-1
        contribute the tail base rate escalated with the same (1+u)^(y-y_j)
        multiplier that the baseline anchor established.
        """
        for name in ("A", "B"):
            b = switching_result.query(
                "scenario == 'baseline' and project_name == @name"
            ).iloc[0]
            t = switching_result.query(
                "scenario == 'all_eor' and project_name == @name"
            ).iloc[0]
            s = switching_result.query(
                "scenario == 'switching' and project_name == @name"
            ).iloc[0]
            tail_nominal = nominal_total(
                annual_cost=t["million_usd_per_yr_oil_revenue"],
                project_start_year=int(b["project_start"]) + D,
                project_duration=L - D,
                escalation_year=int(b["project_start"]) + 1,
                inflation_rate=switching_config.inflation_rate,
            )
            expected = b["nominal_million_usd_oil_revenue"] + tail_nominal
            assert s["lifetime_nominal_million_usd_oil_revenue"] == pytest.approx(
                expected, rel=1e-9
            )

    # ---- #10 ----------------------------------------------------------
    def test_lifetime_pv_oil_revenue_matches_baseline_plus_tail_pv(
        self, switching_result, switching_config
    ):
        for name in ("A", "B"):
            b = switching_result.query(
                "scenario == 'baseline' and project_name == @name"
            ).iloc[0]
            t = switching_result.query(
                "scenario == 'all_eor' and project_name == @name"
            ).iloc[0]
            s = switching_result.query(
                "scenario == 'switching' and project_name == @name"
            ).iloc[0]
            tail_pv = present_value(
                annual_cost=t["million_usd_per_yr_oil_revenue"],
                project_start_year=int(b["project_start"]) + D,
                project_duration=L - D,
                discount_rate=switching_config.discount_rate,
                base_year=switching_config.pv_valuation_year,
                escalation_year=int(b["project_start"]) + 1,
                inflation_rate=switching_config.inflation_rate,
            )
            expected = b["pv_million_usd_oil_revenue"] + tail_pv
            assert s["lifetime_pv_million_usd_oil_revenue"] == pytest.approx(
                expected, rel=1e-9
            )

    # ---- #11 ----------------------------------------------------------
    @pytest.mark.parametrize("col", TIME_WEIGHTED_COLS)
    def test_per_yr_columns_are_time_weighted_mean(self, switching_result, col):
        for name in ("A", "B"):
            b = switching_result.query(
                "scenario == 'baseline' and project_name == @name"
            ).iloc[0]
            t = switching_result.query(
                "scenario == 'all_eor' and project_name == @name"
            ).iloc[0]
            s = switching_result.query(
                "scenario == 'switching' and project_name == @name"
            ).iloc[0]
            expected = (D * b[col] + (L - D) * t[col]) / L
            assert s[col] == pytest.approx(expected, rel=1e-9)

    # ---- #12 ----------------------------------------------------------
    def test_eor_fraction_is_time_weighted(self, switching_result):
        # Facility A: baseline=1.0, all_eor=1.0 → switching=1.0.
        # Facility B: baseline=0.0, all_eor=1.0 → switching=(0.0*12 + 1.0*18)/30 = 0.6.
        sw = switching_result[switching_result["scenario"] == "switching"].set_index(
            "project_name"
        )
        assert sw.loc["A", "eor_fraction"] == pytest.approx(1.0, rel=1e-9)
        assert sw.loc["B", "eor_fraction"] == pytest.approx(0.6, rel=1e-9)

    # ---- #13 ----------------------------------------------------------
    def test_handles_missing_baseline_scenario(
        self, switching_input_frame, switching_config, caplog
    ):
        no_baseline = switching_input_frame[
            switching_input_frame["scenario"] != "baseline"
        ].copy()
        with caplog.at_level(logging.WARNING):
            out = _append_switching_scenario(no_baseline, switching_config)
        pd.testing.assert_frame_equal(
            out.reset_index(drop=True),
            no_baseline.reset_index(drop=True),
        )
        assert any("baseline" in rec.getMessage().lower() for rec in caplog.records)

    # ---- #14 ----------------------------------------------------------
    def test_handles_missing_tail_scenario(
        self, switching_input_frame, switching_config, caplog
    ):
        no_tail = switching_input_frame[
            switching_input_frame["scenario"] != "all_eor"
        ].copy()
        with caplog.at_level(logging.WARNING):
            out = _append_switching_scenario(no_tail, switching_config)
        pd.testing.assert_frame_equal(
            out.reset_index(drop=True),
            no_tail.reset_index(drop=True),
        )
        assert any("all_eor" in rec.getMessage().lower() for rec in caplog.records)

    # ---- #15 ----------------------------------------------------------
    def test_zero_tail_edge_case(self, switching_input_frame, switching_config):
        """When facility_lifetime_years == project_duration (L == D), the
        tail period has zero years; switching rows should equal baseline for
        every exact-sum lifetime column and the PV of oil revenue."""
        cfg = replace(switching_config, facility_lifetime_years=D)
        out = _append_switching_scenario(switching_input_frame, cfg)
        for name in ("A", "B"):
            b = out.query("scenario == 'baseline' and project_name == @name").iloc[0]
            s = out.query("scenario == 'switching' and project_name == @name").iloc[0]
            assert s["lifetime_mtco2e"] == pytest.approx(
                D * b["mtco2e_per_yr_emissions_including_ccs"], rel=1e-9
            )
            assert s["lifetime_million_barrels_oil"] == pytest.approx(
                D * b["million_barrels_oil_per_yr"], rel=1e-9
            )
            assert s["lifetime_pv_million_usd_oil_revenue"] == pytest.approx(
                b["pv_million_usd_oil_revenue"], rel=1e-9
            )


# ---------------------------------------------------------------------------
# Integration tests #16-20 against the real regenerated CSV
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SCENARIOS_CSV = DATA_DIR / "ccs_scenarios_out.csv"


@pytest.fixture(scope="module")
def pipeline_scenarios_df():
    """Full post-pipeline scenarios frame from the regenerated CSV."""
    if not SCENARIOS_CSV.exists():
        pytest.skip(f"{SCENARIOS_CSV} not present; regenerate the pipeline first.")
    return pd.read_csv(SCENARIOS_CSV, encoding="latin-1")


class TestSwitchingIntegration:
    # ---- #16 ----------------------------------------------------------
    def test_switching_rows_present_in_pipeline_output(self, pipeline_scenarios_df):
        assert "switching" in pipeline_scenarios_df["scenario"].unique()
        counts = pipeline_scenarios_df.groupby("scenario").size()
        assert counts["switching"] == counts["baseline"]

    # ---- #17 ----------------------------------------------------------
    @pytest.mark.parametrize(
        "col",
        [
            "lifetime_million_barrels_oil",
            "lifetime_nominal_million_usd_oil_revenue",
            "lifetime_pv_million_usd_oil_revenue",
        ],
    )
    def test_lifetime_oil_columns_on_every_row(self, pipeline_scenarios_df, col):
        assert col in pipeline_scenarios_df.columns
        s = pipeline_scenarios_df[col]
        assert s.notna().all()
        assert (s >= -1e-9).all()  # no negative oil / revenue
        import numpy as np

        assert np.isfinite(s).all()

    # ---- #18 ----------------------------------------------------------
    def test_lifetime_oil_columns_equal_d_year_for_non_switching(
        self, pipeline_scenarios_df
    ):
        # Both `switching` and `switching_from_current` carry L-year oil
        # totals; every other destination scenario only has 12 years of oil.
        non_sw = pipeline_scenarios_df[
            ~pipeline_scenarios_df["scenario"].str.startswith("switching")
        ]
        assert (
            non_sw["lifetime_million_barrels_oil"]
            - D * non_sw["million_barrels_oil_per_yr"]
        ).abs().max() < 1e-9
        assert (
            non_sw["lifetime_nominal_million_usd_oil_revenue"]
            - non_sw["nominal_million_usd_oil_revenue"]
        ).abs().max() < 1e-9
        assert (
            non_sw["lifetime_pv_million_usd_oil_revenue"]
            - non_sw["pv_million_usd_oil_revenue"]
        ).abs().max() < 1e-9

    # ---- #19 ----------------------------------------------------------
    def test_switching_lifetime_matches_exact_two_period_sum_per_facility(
        self, pipeline_scenarios_df
    ):
        """For every (facility × methane_scenario) row in the real pipeline
        output, switching.lifetime_mtco2e equals the exact closed form
        `D × baseline.net_per_yr + (L−D) × all_eor.net_per_yr`. This is the
        per-facility counterpart to the aggregate mass-balance in #22 and
        catches merge-key bugs that would survive an aggregate check."""
        key = ["uid", "project_state", "methane_scenario"]
        per_yr = pipeline_scenarios_df.pivot_table(
            index=key,
            columns="scenario",
            values="mtco2e_per_yr_emissions_including_ccs",
        )
        lifetime = pipeline_scenarios_df.pivot_table(
            index=key, columns="scenario", values="lifetime_mtco2e"
        )
        expected = D * per_yr["baseline"] + (L - D) * per_yr["all_eor"]
        import numpy as np

        diff = (lifetime["switching"] - expected).abs()
        # Allow tiny float noise; use max absolute diff for a tight bound
        # since both sides have the same units (Mt).
        assert np.isfinite(lifetime["switching"]).all()
        assert diff.max() < 1e-9, (
            f"max per-facility abs diff {diff.max():.3g} Mt exceeds tolerance; "
            f"worst offender: {diff.idxmax()}"
        )

    # ---- #20 ----------------------------------------------------------
    def test_switching_eor_fraction_bounded(self, pipeline_scenarios_df):
        """eor_fraction[switching] must satisfy baseline.eor_fraction ≤
        switching.eor_fraction ≤ 1.0 per facility, since the tail is 100% EOR."""
        key = ["uid", "project_state", "methane_scenario"]
        pivot = pipeline_scenarios_df.pivot_table(
            index=key, columns="scenario", values="eor_fraction"
        )
        baseline_eor = pivot["baseline"]
        switching_eor = pivot["switching"]
        assert (switching_eor >= baseline_eor - 1e-9).all()
        assert (switching_eor <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# Test #21 — downstream summarize_scenarios consumer
# ---------------------------------------------------------------------------


class TestSwitchingSummarize:
    def test_summarize_scenario_ids_uses_lifetime_columns_for_switching(
        self, pipeline_scenarios_df
    ):
        from ccs_ghg.plotting import summarize_scenario_ids

        projects_csv = DATA_DIR / "ccs_projects_out.csv"
        if not projects_csv.exists():
            pytest.skip(f"{projects_csv} not present; regenerate the pipeline first.")
        projects_df = pd.read_csv(projects_csv, encoding="latin-1")

        # Restrict to a single methane scenario to keep the comparison 1D.
        one_ch4 = pipeline_scenarios_df[
            pipeline_scenarios_df["methane_scenario"] == "edf20"
        ]
        summary = summarize_scenario_ids(
            one_ch4,
            projects_df=projects_df,
            scenarios=["switching"],
            solar_lcoe_usd_per_mwh=60.0,
        )
        sw_rows = summary[summary["scenario"] == "switching"]
        assert len(sw_rows) == 1

        expected_total_mt = one_ch4[one_ch4["scenario"] == "switching"][
            "lifetime_mtco2e"
        ].sum()
        expected_total_oil = one_ch4[one_ch4["scenario"] == "switching"][
            "lifetime_million_barrels_oil"
        ].sum()
        assert sw_rows["total_mtco2e"].iloc[0] == pytest.approx(
            expected_total_mt, rel=1e-9
        )
        assert sw_rows["total_million_bbls_oil"].iloc[0] == pytest.approx(
            expected_total_oil, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Test #22 — aggregate mass-balance
# ---------------------------------------------------------------------------


class TestSwitchingMassBalance:
    def test_switching_lifetime_mtco2e_aggregate_matches_two_period_sum(
        self, pipeline_scenarios_df
    ):
        """Sum across all facilities under one methane scenario: the
        aggregate switching.lifetime_mtco2e should equal
        D × Σ baseline.net_per_yr + (L−D) × Σ all_eor.net_per_yr, up to
        floating point."""
        key = ["uid", "project_state"]
        for ch4 in pipeline_scenarios_df["methane_scenario"].unique():
            slice_ = pipeline_scenarios_df[
                pipeline_scenarios_df["methane_scenario"] == ch4
            ]
            baseline_per_yr_sum = slice_[slice_["scenario"] == "baseline"][
                "mtco2e_per_yr_emissions_including_ccs"
            ].sum()
            all_eor_per_yr_sum = slice_[slice_["scenario"] == "all_eor"][
                "mtco2e_per_yr_emissions_including_ccs"
            ].sum()
            switching_lifetime_sum = slice_[slice_["scenario"] == "switching"][
                "lifetime_mtco2e"
            ].sum()
            expected = D * baseline_per_yr_sum + (L - D) * all_eor_per_yr_sum
            assert switching_lifetime_sum == pytest.approx(
                expected, rel=1e-6
            ), f"methane_scenario={ch4}: aggregate mismatch"


# ---------------------------------------------------------------------------
# Tests for the second switching variant: `switching_from_current`
# (first period = current_mix instead of baseline).
# ---------------------------------------------------------------------------


class TestSwitchingFromCurrent:
    """Parity coverage for the `switching_from_current` row synthesis.

    Mirrors the key `switching` guarantees for the variant whose first-period
    destination mix is `current_mix` rather than `baseline`. Runs against the
    full pipeline output so it also catches regressions in the
    `_append_switching_scenario` second-call wiring in `run_full_pipeline`.
    """

    def test_rows_present_with_correct_count(self, pipeline_scenarios_df):
        assert "switching_from_current" in pipeline_scenarios_df["scenario"].unique()
        counts = pipeline_scenarios_df.groupby("scenario").size()
        assert counts["switching_from_current"] == counts["current_mix"]

    def test_lifetime_mtco2e_matches_exact_two_period_sum(self, pipeline_scenarios_df):
        """For every (facility × methane) row, `switching_from_current`'s
        `lifetime_mtco2e` should equal D × current_mix_per_yr +
        (L−D) × all_eor_per_yr."""
        key = ["uid", "project_state", "methane_scenario"]
        per_yr = pipeline_scenarios_df.pivot_table(
            index=key,
            columns="scenario",
            values="mtco2e_per_yr_emissions_including_ccs",
        )
        lifetime = pipeline_scenarios_df.pivot_table(
            index=key, columns="scenario", values="lifetime_mtco2e"
        )
        expected = D * per_yr["current_mix"] + (L - D) * per_yr["all_eor"]
        diff = (lifetime["switching_from_current"] - expected).abs()
        assert diff.max() < 1e-9

    def test_45q_credit_copied_from_current_mix(self, pipeline_scenarios_df):
        """45Q runs only during the D-year window at the first-period mix,
        so the credit columns copy from current_mix unchanged."""
        key = ["uid", "project_state", "methane_scenario"]
        pivot = pipeline_scenarios_df.pivot_table(
            index=key, columns="scenario", values="pv_million_usd_credits_45q"
        )
        assert (
            pivot["switching_from_current"] - pivot["current_mix"]
        ).abs().max() < 1e-9

    def test_oil_revenue_pv_covers_full_lifetime(self, pipeline_scenarios_df):
        """Oil flows for all L years under switching_from_current (current_mix
        for years 1..D, 100% EOR for years D+1..L), so the `nominal` and
        `pv` oil-revenue columns must equal the `lifetime_*` counterparts
        on the switching_from_current rows."""
        sw = pipeline_scenarios_df[
            pipeline_scenarios_df["scenario"] == "switching_from_current"
        ]
        assert (
            sw["nominal_million_usd_oil_revenue"]
            - sw["lifetime_nominal_million_usd_oil_revenue"]
        ).abs().max() < 1e-9
        assert (
            sw["pv_million_usd_oil_revenue"] - sw["lifetime_pv_million_usd_oil_revenue"]
        ).abs().max() < 1e-9

    def test_eor_fraction_bounded_between_current_mix_and_one(
        self, pipeline_scenarios_df
    ):
        """switching_from_current's eor_fraction is the time-weighted average
        of current_mix (first-period) and 1.0 (all-EOR tail), so it must sit
        between current_mix.eor_fraction and 1.0 per facility."""
        key = ["uid", "project_state", "methane_scenario"]
        pivot = pipeline_scenarios_df.pivot_table(
            index=key, columns="scenario", values="eor_fraction"
        )
        current_mix_eor = pivot["current_mix"]
        switching_eor = pivot["switching_from_current"]
        assert (switching_eor >= current_mix_eor - 1e-9).all()
        assert (switching_eor <= 1.0 + 1e-9).all()

    def test_lifetime_emissions_higher_than_switching(self, pipeline_scenarios_df):
        """Under any methane scenario where current_mix emits MORE per year
        than baseline (its 92% EOR vs baseline's 17% EOR), the two switching
        variants share the same tail but differ in the first period, so
        switching_from_current should have strictly higher aggregate lifetime
        emissions than plain switching. True for every methane scenario in
        the current configuration."""
        for ch4 in pipeline_scenarios_df["methane_scenario"].unique():
            slice_ = pipeline_scenarios_df[
                pipeline_scenarios_df["methane_scenario"] == ch4
            ]
            sw = slice_[slice_["scenario"] == "switching"]["lifetime_mtco2e"].sum()
            sfc = slice_[slice_["scenario"] == "switching_from_current"][
                "lifetime_mtco2e"
            ].sum()
            assert sfc > sw, (
                f"methane_scenario={ch4}: switching_from_current ({sfc:.1f}) "
                f"should exceed switching ({sw:.1f})"
            )
