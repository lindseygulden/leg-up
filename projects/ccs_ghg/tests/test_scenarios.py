"""Unit tests for the CCS scenarios computation functions.

Run with: pytest projects/ccs_ghg/tests/test_scenarios.py
"""

from dataclasses import replace

import pandas as pd
import pytest

from projects.ccs_ghg.emissions_scenarios.config import (
    DestinationMix,
    MethaneScenario,
    ScenarioConfig,
)
from projects.ccs_ghg.emissions_scenarios.scenarios import (
    compute_project_emissions,
    compute_scenario_emissions_factors,
    compute_scenario_summary,
    load_industry_emissions,
    merge_project_factors,
    present_value,
)

# ---------------------------------------------------------------------------
# Present value
# ---------------------------------------------------------------------------


class TestPresentValue:
    def test_single_project(self):
        """One year, no discounting offset -> just annual_cost / (1+r)^1."""
        pv = present_value(
            annual_cost=100.0,
            project_start_year=2025,
            project_duration=1,
            discount_rate=0.05,
            base_year=2025,
        )
        expected = 100.0 / 1.05
        assert pv == pytest.approx(expected, rel=1e-9)

    def test_zero_discount_rate(self):
        """With r=0, PV = annual_cost x duration."""
        pv = present_value(
            annual_cost=50.0,
            project_start_year=2025,
            project_duration=10,
            discount_rate=0.0,
            base_year=2025,
        )
        assert pv == pytest.approx(500.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Scenario emissions factors
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_emissions_factors():
    """Minimal per-(source, destination) emissions factors DataFrame."""
    return pd.DataFrame(
        {
            "co2_source": ["hydrogen", "hydrogen", "ngp", "ngp"],
            "co2_destination": ["eor", "gs", "eor", "gs"],
            "tco2e_emitted_per_tco2_ccs": [0.5, 0.3, 0.6, 0.4],
            "tco2_emitted_per_tco2_ccs": [0.4, 0.2, 0.5, 0.3],
            "net_tco2e_emitted_per_tco2_ccs": [-0.5, -0.7, -0.4, -0.6],
            "net_tco2_emitted_per_tco2_ccs": [-0.6, -0.8, -0.5, -0.7],
        }
    )


@pytest.fixture
def minimal_scenario_config():
    """ScenarioConfig with just enough fields for compute_scenario_emissions_factors."""
    return ScenarioConfig(
        emissions_for_industries_path="dummy",
        projects_info_path="dummy",
        ccs_emissions_factors_path="dummy",
        industry_emissions_cols=[],
        industry_emissions_which_flavor=[],
        facility_keep_cols=[],
        projects_merge_on="product",
        ccs_emissions_merge_on="co2_source",
        nameplate_ccs_col="nameplate",
        project_start_col="start",
        project_unabated_mtco2e_col="unabated",
        product_col_name="product",
        industrial_product_with_detail_col_name="industrial_product",
        usd_per_tco2_45q_eor=85,
        usd_per_tco2_45q_gs=85,
        inflation_rate=0.0,
        adjusted_capt_frac_col_name="frac",
        weighted_scenarios=[
            DestinationMix("current_mix", {"eor": 80, "gs": 20}),
            DestinationMix("all_eor", {"eor": 100, "gs": 0}),
            DestinationMix("all_gs", {"eor": 0, "gs": 100}),
        ],
    )


class TestComputeScenarioEmissionsFactors:
    def test_output_has_emissions_scenario_column(
        self, sample_emissions_factors, minimal_scenario_config
    ):
        result = compute_scenario_emissions_factors(
            sample_emissions_factors, minimal_scenario_config
        )
        assert "scenario" in result.columns

    def test_output_scenarios(self, sample_emissions_factors, minimal_scenario_config):
        result = compute_scenario_emissions_factors(
            sample_emissions_factors, minimal_scenario_config
        )
        scenarios = set(result["scenario"].unique())
        assert scenarios == {
            "current_mix",
            "all_eor",
            "all_gs",
            "baseline",
            "nameplate_implied",
            "nameplate",
        }

    def test_output_grain(self, sample_emissions_factors, minimal_scenario_config):
        """Each (source, destination, scenario) combination should appear once."""
        result = compute_scenario_emissions_factors(
            sample_emissions_factors, minimal_scenario_config
        )
        grain_cols = ["co2_source", "stated_co2_destination", "scenario"]
        assert not result.duplicated(subset=grain_cols).any()

    def test_all_eor_uses_eor_factors(
        self, sample_emissions_factors, minimal_scenario_config
    ):
        """all_eor scenario should use the eor factor for each source."""
        result = compute_scenario_emissions_factors(
            sample_emissions_factors, minimal_scenario_config
        )
        all_eor = result.loc[result["scenario"] == "all_eor"]
        # For hydrogen, eor factor is 0.5 — all_eor should give exactly that
        h2_rows = all_eor.loc[all_eor["co2_source"] == "hydrogen"]
        assert h2_rows["tco2e_emitted_per_tco2_ccs"].iloc[0] == pytest.approx(
            0.5, rel=1e-9
        )

    def test_all_gs_uses_gs_factors(
        self, sample_emissions_factors, minimal_scenario_config
    ):
        result = compute_scenario_emissions_factors(
            sample_emissions_factors, minimal_scenario_config
        )
        all_gs = result.loc[result["scenario"] == "all_gs"]
        h2_rows = all_gs.loc[all_gs["co2_source"] == "hydrogen"]
        assert h2_rows["tco2e_emitted_per_tco2_ccs"].iloc[0] == pytest.approx(
            0.3, rel=1e-9
        )

    def test_current_mix_weighted(
        self, sample_emissions_factors, minimal_scenario_config
    ):
        """current_mix with 80/20 eor/gs should be (80*eor + 20*gs)/100."""
        result = compute_scenario_emissions_factors(
            sample_emissions_factors, minimal_scenario_config
        )
        cm = result.loc[result["scenario"] == "current_mix"]
        h2_rows = cm.loc[cm["co2_source"] == "hydrogen"]
        expected = (80 * 0.5 + 20 * 0.3) / 100  # 0.46
        assert h2_rows["tco2e_emitted_per_tco2_ccs"].iloc[0] == pytest.approx(
            expected, rel=1e-9
        )

    def test_unabated_matches_input(
        self, sample_emissions_factors, minimal_scenario_config
    ):
        """Unabated rows should match the per-(source, destination) input factors."""
        result = compute_scenario_emissions_factors(
            sample_emissions_factors, minimal_scenario_config
        )
        unabated = result.loc[result["scenario"] == "baseline"].sort_values(
            ["co2_source", "stated_co2_destination"]
        )
        expected = sample_emissions_factors.sort_values(
            ["co2_source", "co2_destination"]
        )
        for col in ["tco2e_emitted_per_tco2_ccs", "tco2_emitted_per_tco2_ccs"]:
            assert list(unabated[col]) == pytest.approx(list(expected[col]), rel=1e-9)

    def test_has_stated_and_assumed_destination_columns(
        self, sample_emissions_factors, minimal_scenario_config
    ):
        result = compute_scenario_emissions_factors(
            sample_emissions_factors, minimal_scenario_config
        )
        assert "stated_co2_destination" in result.columns
        assert "assumed_co2_destination" in result.columns
        assert "co2_destination" not in result.columns

    def test_assumed_destination_weighted_scenario(
        self, sample_emissions_factors, minimal_scenario_config
    ):
        result = compute_scenario_emissions_factors(
            sample_emissions_factors, minimal_scenario_config
        )
        cm = result.loc[result["scenario"] == "current_mix"]
        assert (cm["assumed_co2_destination"] == "80% eor/20% gs").all()

    def test_assumed_destination_baseline(
        self, sample_emissions_factors, minimal_scenario_config
    ):
        result = compute_scenario_emissions_factors(
            sample_emissions_factors, minimal_scenario_config
        )
        bl = result.loc[result["scenario"] == "baseline"]
        assert (bl["assumed_co2_destination"] == bl["stated_co2_destination"]).all()

    def test_nameplate_implied_zero_factors(
        self, sample_emissions_factors, minimal_scenario_config
    ):
        result = compute_scenario_emissions_factors(
            sample_emissions_factors, minimal_scenario_config
        )
        np_rows = result.loc[result["scenario"] == "nameplate_implied"]
        factor_cols = [
            "tco2e_emitted_per_tco2_ccs",
            "tco2_emitted_per_tco2_ccs",
            "net_tco2e_emitted_per_tco2_ccs",
            "net_tco2_emitted_per_tco2_ccs",
        ]
        for col in factor_cols:
            assert (np_rows[col] == 0.0).all()
        assert (np_rows["assumed_co2_destination"] == "as stated").all()

    def test_nameplate_excluded_when_disabled(
        self, sample_emissions_factors, minimal_scenario_config
    ):
        minimal_scenario_config.include_nameplate_case = False
        result = compute_scenario_emissions_factors(
            sample_emissions_factors, minimal_scenario_config
        )
        assert "nameplate_implied" not in result["scenario"].values


# ---------------------------------------------------------------------------
# LeakRateScenario config parsing
# ---------------------------------------------------------------------------


class TestMethaneScenario:
    def test_from_dict(self):
        ms = MethaneScenario.from_dict(
            {"name": "low", "gwp_ch4": 29.8, "methane_leak_rate": 0.02}
        )
        assert ms.name == "low"
        assert ms.gwp_ch4 == 29.8
        assert ms.methane_leak_rate == 0.02

    def test_scenario_config_parses_methane_scenarios(self):
        cfg = ScenarioConfig.from_dict(
            {
                "emissions_for_industries_path": "dummy",
                "projects_info_path": "dummy",
                "ccs_emissions_factors_path": "dummy",
                "industry_emissions_cols": [],
                "industry_emissions_which_flavor": [],
                "facility_keep_cols": [],
                "projects_merge_on": "x",
                "ccs_emissions_merge_on": "x",
                "nameplate_ccs_col": "x",
                "project_start_col": "x",
                "project_unabated_mtco2e_col": "x",
                "product_col_name": "x",
                "industrial_product_with_detail_col_name": "x",
                "usd_per_tco2_45q_eor": 85,
                "usd_per_tco2_45q_gs": 85,
                "adjusted_capt_frac_col_name": "x",
                "use_precomputed": False,
                "methane_scenarios": [
                    {"name": "base", "gwp_ch4": 86, "methane_leak_rate": 0.035},
                    {"name": "low", "gwp_ch4": 29.8, "methane_leak_rate": 0.02},
                ],
            }
        )
        assert cfg.use_precomputed is False
        assert len(cfg.methane_scenarios) == 2
        assert cfg.methane_scenarios[1].methane_leak_rate == 0.02
        assert cfg.methane_scenarios[1].gwp_ch4 == 29.8


# ---------------------------------------------------------------------------
# load_industry_emissions with pre-computed DataFrame
# ---------------------------------------------------------------------------


class TestLoadIndustryEmissionsPrecomputed:
    def test_accepts_precomputed_dataframe(self, minimal_scenario_config):
        minimal_scenario_config.industry_emissions_which_flavor = ["hydrogen"]
        minimal_scenario_config.industrial_product_with_detail_col_name = (
            "industrial_product"
        )
        minimal_scenario_config.product_col_name = "simple_product_name"
        minimal_scenario_config.adjusted_capt_frac_col_name = (
            "adjusted_capture_fraction_tco2e"
        )
        minimal_scenario_config.industry_emissions_cols = [
            "industrial_product",
            "simple_product_name",
            "adjusted_capture_fraction_tco2e",
        ]
        precomputed = pd.DataFrame(
            {
                "industrial_product": ["hydrogen", "ammonia"],
                "simple_product_name": ["hydrogen", "ammonia"],
                "adjusted_capture_fraction_tco2e": [0.45, 0.55],
            }
        )
        result = load_industry_emissions(
            minimal_scenario_config, industry_df=precomputed
        )
        assert len(result) == 1
        assert result.iloc[0]["product"] == "hydrogen"
        assert result.iloc[0]["adjusted_capture_fraction_tco2e"] == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Oil production and scenario summary
# ---------------------------------------------------------------------------


@pytest.fixture
def oil_scenario_config():
    """Config with weighted scenarios for oil production tests."""
    return ScenarioConfig(
        emissions_for_industries_path="dummy",
        projects_info_path="dummy",
        ccs_emissions_factors_path="dummy",
        industry_emissions_cols=[],
        industry_emissions_which_flavor=["hydrogen"],
        facility_keep_cols=[
            "project_name",
            "carbon_fate",
            "product",
            "nameplate_captured_mtco2_per_yr",
            "unabated_mtco2_per_yr",
            "unabated_mtco2e_per_yr",
            "project_start",
        ],
        projects_merge_on=["product", "carbon_fate"],
        ccs_emissions_merge_on=["product", "stated_co2_destination"],
        nameplate_ccs_col="nameplate_captured_mtco2_per_yr",
        project_start_col="project_start",
        project_unabated_mtco2e_col="unabated_mtco2e_per_yr",
        product_col_name="simple_product_name",
        industrial_product_with_detail_col_name="industrial_product",
        usd_per_tco2_45q_eor=85,
        usd_per_tco2_45q_gs=85,
        inflation_rate=0.0,
        adjusted_capt_frac_col_name="adjusted_capture_fraction_tco2e",
        recovery_factor_bbl_per_tco2=3.0,
        weighted_scenarios=[
            DestinationMix("current_mix", {"eor": 80, "gs": 20}),
            DestinationMix("all_eor", {"eor": 100, "gs": 0}),
            DestinationMix("all_gs", {"eor": 0, "gs": 100}),
        ],
    )


@pytest.fixture
def oil_test_projects():
    return pd.DataFrame(
        {
            "project_name": ["Project A", "Project B"],
            "carbon_fate": ["eor", "gs"],
            "product": ["hydrogen", "hydrogen"],
            "nameplate_captured_mtco2_per_yr": [2.0, 3.0],
            "unabated_mtco2_per_yr": [5.0, 8.0],
            "unabated_mtco2e_per_yr": [5.0, 8.0],
            "project_start": [2027, 2028],
        }
    )


@pytest.fixture
def oil_test_industry():
    return pd.DataFrame(
        {
            "industrial_product": ["hydrogen"],
            "simple_product_name": ["hydrogen"],
            "adjusted_capture_fraction_tco2": [0.5],
            "adjusted_capture_fraction_tco2e": [0.5],
        }
    )


@pytest.fixture
def oil_test_result(
    oil_test_projects,
    oil_test_industry,
    oil_scenario_config,
    sample_emissions_factors,
):
    scenario_factors = compute_scenario_emissions_factors(
        sample_emissions_factors, oil_scenario_config
    )
    industry = load_industry_emissions(
        oil_scenario_config, industry_df=oil_test_industry
    )
    merged = merge_project_factors(
        oil_test_projects, scenario_factors, industry, oil_scenario_config
    )
    return compute_project_emissions(merged, oil_scenario_config)


class TestOilProduction:
    def test_eor_fraction_weighted_scenario(self, oil_test_result):
        cm = oil_test_result[oil_test_result["scenario"] == "current_mix"]
        assert (cm["eor_fraction"] == 0.8).all()

    def test_eor_fraction_all_eor(self, oil_test_result):
        ae = oil_test_result[oil_test_result["scenario"] == "all_eor"]
        assert (ae["eor_fraction"] == 1.0).all()

    def test_eor_fraction_all_gs(self, oil_test_result):
        gs = oil_test_result[oil_test_result["scenario"] == "all_gs"]
        assert (gs["eor_fraction"] == 0.0).all()

    def test_eor_fraction_baseline_uses_stated(self, oil_test_result):
        bl = oil_test_result[oil_test_result["scenario"] == "baseline"]
        eor_row = bl[bl["stated_co2_destination"] == "eor"]
        gs_row = bl[bl["stated_co2_destination"] == "gs"]
        assert (eor_row["eor_fraction"] == 1.0).all()
        assert (gs_row["eor_fraction"] == 0.0).all()

    def test_barrels_equals_source_capture_times_fraction_times_recovery(
        self, oil_test_result
    ):
        """With `use_operator_nameplate_when_available=False` (default), oil
        barrels uses the implied capture as basis (unabated × capture_fraction),
        not the operator-supplied nameplate."""
        for _, row in oil_test_result.iterrows():
            source = (
                row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            )
            expected = source * row["eor_fraction"] * 3.0
            assert row["million_barrels_oil_per_yr"] == pytest.approx(
                expected, rel=1e-9
            )

    def test_all_gs_zero_barrels(self, oil_test_result):
        gs = oil_test_result[oil_test_result["scenario"] == "all_gs"]
        assert (gs["million_barrels_oil_per_yr"] == 0.0).all()

    def test_current_mix_barrels(self, oil_test_result):
        cm = oil_test_result[oil_test_result["scenario"] == "current_mix"]
        for _, row in cm.iterrows():
            source = (
                row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            )
            expected = source * 0.8 * 3.0
            assert row["million_barrels_oil_per_yr"] == pytest.approx(
                expected, rel=1e-9
            )


class TestScenarioSummary:
    def test_summary_has_expected_columns(self, oil_test_result):
        summary = compute_scenario_summary(oil_test_result)
        for col in [
            "scenario",
            "total_nameplate_mtco2_per_yr",
            "total_million_barrels_oil_per_yr",
            "total_avg_million_usd_per_yr_45q",
            "total_pv_million_usd_credits_45q",
            "total_mtco2e_per_yr_emissions",
        ]:
            assert col in summary.columns, f"Missing column: {col}"

    def test_summary_one_row_per_scenario(self, oil_test_result):
        summary = compute_scenario_summary(oil_test_result)
        assert len(summary) == len(oil_test_result["scenario"].unique())

    def test_all_gs_zero_total_barrels(self, oil_test_result):
        summary = compute_scenario_summary(oil_test_result)
        gs_row = summary[summary["scenario"] == "all_gs"]
        assert gs_row["total_million_barrels_oil_per_yr"].iloc[0] == 0.0

    def test_subsidy_same_across_scenarios(self, oil_test_result):
        summary = compute_scenario_summary(oil_test_result)
        subsidies = summary["total_avg_million_usd_per_yr_45q"].unique()
        assert len(subsidies) == 1

    def test_total_barrels_all_eor_gt_current_mix(self, oil_test_result):
        summary = compute_scenario_summary(oil_test_result)
        ae = summary.loc[
            summary["scenario"] == "all_eor", "total_million_barrels_oil_per_yr"
        ].iloc[0]
        cm = summary.loc[
            summary["scenario"] == "current_mix", "total_million_barrels_oil_per_yr"
        ].iloc[0]
        assert ae > cm

    def test_summary_groups_by_methane_scenario(self, oil_test_result):
        df = oil_test_result.copy()
        df["methane_scenario"] = "high_gwp"
        low = oil_test_result.copy()
        low["methane_scenario"] = "low_gwp"
        combined = pd.concat([df, low], ignore_index=True)
        summary = compute_scenario_summary(combined)
        assert "methane_scenario" in summary.columns
        n_scenarios = combined["scenario"].nunique()
        assert len(summary) == n_scenarios * 2


# ---------------------------------------------------------------------------
# Coverage gap tests
# ---------------------------------------------------------------------------


class TestPresentValueBaseYearNone:
    def test_base_year_defaults_to_start_year(self):
        pv_explicit = present_value(
            annual_cost=100.0,
            project_start_year=2030,
            project_duration=5,
            discount_rate=0.05,
            base_year=2030,
        )
        pv_none = present_value(
            annual_cost=100.0,
            project_start_year=2030,
            project_duration=5,
            discount_rate=0.05,
            base_year=None,
        )
        assert pv_none == pytest.approx(pv_explicit, rel=1e-9)


class TestZeroWeightScenarioSkipped:
    def test_zero_weight_scenario_excluded(self, sample_emissions_factors):
        cfg = ScenarioConfig(
            emissions_for_industries_path="dummy",
            projects_info_path="dummy",
            ccs_emissions_factors_path="dummy",
            industry_emissions_cols=[],
            industry_emissions_which_flavor=[],
            facility_keep_cols=[],
            projects_merge_on="product",
            ccs_emissions_merge_on="co2_source",
            weighted_scenarios=[
                DestinationMix("zero_mix", {"eor": 0, "gs": 0}),
                DestinationMix("all_gs", {"eor": 0, "gs": 100}),
            ],
        )
        result = compute_scenario_emissions_factors(sample_emissions_factors, cfg)
        assert "zero_mix" not in result["scenario"].values
        assert "all_gs" in result["scenario"].values


class TestLoadIndustryEmissionsFromCSV:
    def test_reads_from_csv_when_no_dataframe_provided(self, tmp_path):
        csv_path = str(tmp_path / "industry.csv")
        pd.DataFrame(
            {
                "industrial_product": ["hydrogen", "ammonia"],
                "simple_product_name": ["hydrogen", "ammonia"],
                "adjusted_capture_fraction_tco2e": [0.45, 0.55],
                "adjusted_capture_fraction_tco2": [0.50, 0.60],
            }
        ).to_csv(csv_path, index=False)
        cfg = ScenarioConfig(
            emissions_for_industries_path=csv_path,
            projects_info_path="dummy",
            ccs_emissions_factors_path="dummy",
            industry_emissions_cols=[
                "industrial_product",
                "simple_product_name",
                "adjusted_capture_fraction_tco2e",
                "adjusted_capture_fraction_tco2",
            ],
            industry_emissions_which_flavor=["hydrogen"],
            facility_keep_cols=[],
            projects_merge_on="product",
            ccs_emissions_merge_on="co2_source",
            product_col_name="simple_product_name",
            industrial_product_with_detail_col_name="industrial_product",
            adjusted_capt_frac_col_name="adjusted_capture_fraction_tco2e",
        )
        result = load_industry_emissions(cfg)
        assert len(result) == 1
        assert result.iloc[0]["product"] == "hydrogen"


class TestOperatorNameplateSubsidy:
    """`use_operator_nameplate_when_available` now picks the "source of truth"
    capture basis for baseline/weighted scenarios, 45Q, and oil accounting."""

    def _run(self, oil_test_projects, oil_test_industry, sample_emissions_factors, cfg):
        scenario_factors = compute_scenario_emissions_factors(
            sample_emissions_factors, cfg
        )
        industry = load_industry_emissions(cfg, industry_df=oil_test_industry)
        merged = merge_project_factors(
            oil_test_projects, scenario_factors, industry, cfg
        )
        return compute_project_emissions(merged, cfg)

    def test_subsidy_uses_implied_when_flag_false(
        self,
        oil_test_projects,
        oil_test_industry,
        oil_scenario_config,
        sample_emissions_factors,
    ):
        cfg = replace(oil_scenario_config, use_operator_nameplate_when_available=False)
        result = self._run(
            oil_test_projects, oil_test_industry, sample_emissions_factors, cfg
        )
        for _, row in result.iterrows():
            implied = (
                row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            )
            expected = implied * 85
            assert row["avg_million_usd_per_yr_45q"] == pytest.approx(
                expected, rel=1e-9
            )

    def test_subsidy_uses_effective_nameplate_when_flag_true(
        self,
        oil_test_projects,
        oil_test_industry,
        oil_scenario_config,
        sample_emissions_factors,
    ):
        cfg = replace(oil_scenario_config, use_operator_nameplate_when_available=True)
        result = self._run(
            oil_test_projects, oil_test_industry, sample_emissions_factors, cfg
        )
        for _, row in result.iterrows():
            implied = (
                row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            )
            nameplate = row.get("nameplate_captured_mtco2_per_yr")
            effective = nameplate if pd.notna(nameplate) else implied
            expected = effective * 85
            assert row["avg_million_usd_per_yr_45q"] == pytest.approx(
                expected, rel=1e-9
            )


class TestBlended45QRate:
    """Verify that different EOR/GS 45Q rates are correctly blended."""

    def _run(self, oil_test_projects, oil_test_industry, sample_emissions_factors, cfg):
        scenario_factors = compute_scenario_emissions_factors(
            sample_emissions_factors, cfg
        )
        industry = load_industry_emissions(cfg, industry_df=oil_test_industry)
        merged = merge_project_factors(
            oil_test_projects, scenario_factors, industry, cfg
        )
        return compute_project_emissions(merged, cfg)

    def test_all_eor_uses_eor_rate(
        self,
        oil_test_projects,
        oil_test_industry,
        oil_scenario_config,
        sample_emissions_factors,
    ):
        cfg = replace(
            oil_scenario_config, usd_per_tco2_45q_eor=60, usd_per_tco2_45q_gs=85
        )
        result = self._run(
            oil_test_projects, oil_test_industry, sample_emissions_factors, cfg
        )
        ae = result[result["scenario"] == "all_eor"]
        for _, row in ae.iterrows():
            basis = row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            expected = basis * 60
            assert row["avg_million_usd_per_yr_45q"] == pytest.approx(
                expected, rel=1e-9
            )

    def test_all_gs_uses_gs_rate(
        self,
        oil_test_projects,
        oil_test_industry,
        oil_scenario_config,
        sample_emissions_factors,
    ):
        cfg = replace(
            oil_scenario_config, usd_per_tco2_45q_eor=60, usd_per_tco2_45q_gs=85
        )
        result = self._run(
            oil_test_projects, oil_test_industry, sample_emissions_factors, cfg
        )
        gs = result[result["scenario"] == "all_gs"]
        for _, row in gs.iterrows():
            basis = row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            expected = basis * 85
            assert row["avg_million_usd_per_yr_45q"] == pytest.approx(
                expected, rel=1e-9
            )

    def test_current_mix_blends_rates(
        self,
        oil_test_projects,
        oil_test_industry,
        oil_scenario_config,
        sample_emissions_factors,
    ):
        cfg = replace(
            oil_scenario_config, usd_per_tco2_45q_eor=60, usd_per_tco2_45q_gs=85
        )
        result = self._run(
            oil_test_projects, oil_test_industry, sample_emissions_factors, cfg
        )
        cm = result[result["scenario"] == "current_mix"]
        for _, row in cm.iterrows():
            basis = row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            blended_rate = 0.8 * 60 + 0.2 * 85  # 80% EOR, 20% GS
            expected = basis * blended_rate
            assert row["avg_million_usd_per_yr_45q"] == pytest.approx(
                expected, rel=1e-9
            )

    def test_baseline_uses_stated_destination(
        self,
        oil_test_projects,
        oil_test_industry,
        oil_scenario_config,
        sample_emissions_factors,
    ):
        cfg = replace(
            oil_scenario_config, usd_per_tco2_45q_eor=60, usd_per_tco2_45q_gs=85
        )
        result = self._run(
            oil_test_projects, oil_test_industry, sample_emissions_factors, cfg
        )
        bl = result[result["scenario"] == "baseline"]
        for _, row in bl.iterrows():
            basis = row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            if row["stated_co2_destination"] == "eor":
                expected = basis * 60
            else:
                expected = basis * 85
            assert row["avg_million_usd_per_yr_45q"] == pytest.approx(
                expected, rel=1e-9
            )

    def test_equal_rates_match_old_behavior(
        self,
        oil_test_projects,
        oil_test_industry,
        oil_scenario_config,
        sample_emissions_factors,
    ):
        cfg = replace(
            oil_scenario_config, usd_per_tco2_45q_eor=85, usd_per_tco2_45q_gs=85
        )
        result = self._run(
            oil_test_projects, oil_test_industry, sample_emissions_factors, cfg
        )
        for _, row in result.iterrows():
            if row["project_start"] < cfg.forty_five_q_earliest_start:
                continue
            basis = row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            expected = basis * 85
            assert row["avg_million_usd_per_yr_45q"] == pytest.approx(
                expected, rel=1e-9
            )
