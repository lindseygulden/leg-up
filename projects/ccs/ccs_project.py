"""Builds subclass for project that represents a Carbon Capture and Sequestration project"""

# pylint: disable=too-many-instance-attributes,too-few-public-methods
import logging
from pathlib import PosixPath
from typing import Union
import numpy as np
from projects.ccs.project import Project

logging.basicConfig(level=logging.INFO)


class CCSProject(Project):
    """Defines a Carbon Capture and Storage project; computes key characteristics based on inputs"""

    def __init__(self, params: Union[Union[str, PosixPath], dict]):
        # initialize parent class __init__
        super().__init__(params)
        self.type = "ccs"
        self.industry = self.config["industry"]
        if self.config["cost_method"] == "defined":
            self.capture_cost_usd_per_tco2 = self.config["capture_cost_usd_per_tco2"]
            self.storage_cost_usd_per_tco2 = self.config["storage_cost_usd_per_tco2"]
            self.transport_cost_usd_per_tco2 = self.config[
                "transport_cost_usd_per_tco2"
            ]
        # placeholder for computation engine

        # Tax credit rate for use of co2 for enhanced oil recovery (eor)
        self.eor_credit_per_tco2 = 60
        if "eor_credit_per_tco2" in self.config:
            self.eor_credit_per_tco2 = self.config["eor_credit_per_tco2"]

        # Tax credit rate for use of co2 for geologic storage (gs)
        self.gs_credit_per_tco2 = 85
        if "gs_credit_per_tco2" in self.config:
            self.gs_credit_per_tco2 = self.config["gs_credit_per_tco2"]

        # Assumed EOR recovery factor of 3 bbl oil/tco2 if not specified
        self.recovery_factor_bbl_oil_per_tco2 = 3
        if "recovery_factor_bbl_oil_per_tco2" in self.config:
            self.recovery_factor_bbl_oil_per_tco2 = self.config[
                "recovery_factor_bbl_oil_per_tco2"
            ]

        # If yearly time series of tco2 sequestered is provided, use that as basis for project and project length
        if "tco2_sequestered_per_yr" in self.config:
            self.tco2_sequestered_per_yr = self.config["tco2_sequestered_per_yr"]
            self.total_length_project = len(self.tco2_sequestered_per_yr)
        elif "total_length_project" in self.config:
            self.total_length_project = self.config["total_length_project"]
            self.tco2_sequestered_per_yr = [0] * 3 + [1] * self.total_length_project - 3

        # compute total tons tco2 sequestered over project (for unit-value calculations)
        self.total_tco2 = np.sum(self.tco2_sequestered_per_yr)

        # TODO add error checking/handling for alignment of length of oil prices and tco2
        self.oil_prices = self.config["oil_prices"]
        self.oil_breakeven_price = self.config["oil_breakeven_price"]

        if self.config["revenue_method"] == "defined":
            self.revenue_from_eor_subsidy_usd_per_tco2 = self.config[
                "revenue_from_eor_subsidy_usd_per_tco2"
            ]
            self.revenue_from_oil_sold_usd_per_tco2 = self.config[
                "revenue_from_oil_sold_usd_per_tco2"
            ]
            self.gs_total_unit_revenue_usd_per_tco2 = self.config[
                "gs_total_unit_revenue_usd_per_tco2"
            ]
        elif self.config["revenue_method"] == "computed":
            self._compute_revenue()

        # once we've assigned all variables in the config file, delete the attribute
        delattr(self, "config")

    def _compute_revenue(self):
        """compute annual average discounted revenue"""
        # compute bbl of oil sold if co2 is used for eor
        self.oil_bbl_sold_per_yr = self.unit_conversion(
            self.tco2_sequestered_per_yr, self.recovery_factor_bbl_oil_per_tco2
        )

        # Compute the effective price per bbl (after taking into account the specified breakeven price per bbl)?
        self.oil_effective_price = [
            b - self.oil_breakeven_price for b in self.oil_prices
        ]

        # What is the unit revenue coming from oil production caused by injecting co2 for EOR?
        # (in units of usd per tco2)?
        self.pv_revenue_from_oil_sold_usd = self.pv(
            self.discount_rate - self.inflation_rate,
            self.unit_conversion(self.oil_bbl_sold_per_yr, self.oil_effective_price),
        )

        # What is the unit revenue coming from the 45q payment?
        self.pv_revenue_from_eor_subsidy_usd = self.pv(
            self.discount_rate - self.inflation_rate,
            [
                c * p
                for c, p in zip(
                    self.tco2_sequestered_per_yr,
                    [self.eor_credit_per_tco2] * self.total_length_project,
                )
            ],
        )

        self.eor_total_unit_revenue_usd_per_tco2 = (
            self.pv_revenue_from_eor_subsidy_usd + self.pv_revenue_from_oil_sold_usd
        ) / self.total_tco2
        # compute pv of subsidy payments for eor in units of usd/tco2
        self.eor_subsidy_unit_revenue_usd_per_tco2 = (
            self.pv_revenue_from_eor_subsidy_usd / self.total_tco2
        )
        # compute pv of subsidy payments for gs in units of usd/tco2
        self.pv_gs_total_unit_revenue_usd = self.pv(
            self.discount_rate - self.inflation_rate,
            self.unit_conversion(self.tco2_sequestered_per_yr, self.gs_credit_per_tco2),
        )
        self.gs_subsidy_unit_revenue_usd_per_tco2 = (
            self.pv_gs_total_unit_revenue_usd / self.total_tco2
        )

        self.total_eor_usd_per_tco2 = (
            self.eor_total_unit_revenue_usd_per_tco2
            - self.capture_cost_usd_per_tco2
            - self.transport_cost_usd_per_tco2
            - self.storage_cost_usd_per_tco2
        )

        self.total_gs_usd_per_tco2 = (
            self.gs_subsidy_unit_revenue_usd_per_tco2
            - self.capture_cost_usd_per_tco2
            - self.transport_cost_usd_per_tco2
            - self.storage_cost_usd_per_tco2
        )
