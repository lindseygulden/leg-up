"""Builds subclass for project that represents a rooftop solar project"""

# pylint: disable=too-many-instance-attributes,too-few-public-methods
import logging
from pathlib import PosixPath
from typing import Tuple, Union

import numpy_financial as npf

from projects.ccs.project import Project

logging.basicConfig(level=logging.INFO)


class RooftopSolarProject(Project):
    """Defines a rooftop solar project; computes key characteristics based on inputs"""

    def __init__(self, params: Union[Union[str, PosixPath], dict]):
        # initialize parent class __init__
        super().__init__(params)
        self.type = "rooftop_solar"

        # details about this particular solar array
        self.region = self.config["region"]
        self.state = self.config["state"]
        self.tilt = self.config["tilt"]
        self.azimuth = self.config["azimuth"]
        self.system_size_kw = self.config["kw"]
        self.kwh_per_yr = self.config["kwh_per_yr"]  # from PVwatts

        # details about carbon intensity and price of electricity in this location, cost of PV array install
        self.installation_cost_usd = self.config["installation_cost_usd"]
        self.tco2_per_kwh = self.config["tco2_per_kwh"]
        self.usd_per_kwh = self.config["usd_per_kwh"]

        # compute financials
        self.avg_discounted_unit_value_solar_usd_per_yr = (
            self._avg_discounted_unit_cash_flow(
                [self.kwh_per_yr] * self.project_length_yrs,
                [self.usd_per_kwh] * self.project_length_yrs,
            )
        )
        self.usd_per_yr = self._compute_usd_per_yr()

        # tco2 totals and tco2/yr

        self.tco2_per_yr, self.total_tco2 = self._compute_tco2()
        self.pv_all_solar = npf.npv(
            self.discount_rate - self.inflation_rate,
            [self.usd_per_yr] * self.project_length_yrs,
        )
        self.pv_cost = npf.npv(
            self.discount_rate - self.inflation_rate,
            [self.installation_cost_usd] + [0] * (self.project_length_yrs - 1),
        )

        self.cost_usd_per_tco2 = self.pv_cost / self.total_tco2

        self.avg_discounted_unit_value_solar_usd_per_tco2 = (
            self._compute_unit_value_per_tco2(
                self.avg_discounted_unit_value_solar_usd_per_yr
            )
        )

        # once we've assigned all variables in the config file, delete the attribute
        delattr(self, "config")

    def _compute_usd_per_yr(self) -> float:
        """Computes nominal expenditure in today's USD for value of kwh per year"""
        return self.usd_per_kwh * self.kwh_per_yr

    def _compute_tco2(self) -> Tuple[float, float]:
        """compute tco2 emissions averted because of panels each year and total averted over length of project
        Returns:
            tco2_per_year: the average tco2 emissions averted per year over the lenght of the project
            total_tco2: the total tco2 emissions that are averted over the lifetime of the project
        """
        tco2_per_year = self.tco2_per_kwh * self.kwh_per_yr
        total_tco2 = tco2_per_year * self.project_length_yrs

        return (tco2_per_year, total_tco2)

    def _compute_price_of_tco2_removal(self) -> float:
        """Computes the total cost of tco2 removal"""
        return self.installation_cost_usd / self.total_tco2

    def _compute_unit_value_per_tco2(self, value_usd_per_yr) -> float:
        return value_usd_per_yr / self.tco2_per_yr
