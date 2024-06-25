"""Builds subclass for project that represents a rooftop solar project"""

# pylint: disable=too-many-instance-attributes,too-few-public-methods
import logging
from pathlib import PosixPath
from typing import Tuple, Union

from projects.ccs.project import Project

logging.basicConfig(level=logging.INFO)


class RooftopSolarProject(Project):
    """Defines a rooftop solar project; computes key characteristics based on inputs"""

    def __init__(self, params: Union[Union[str, PosixPath], dict]):
        # initialize parent class __init__
        super().__init__(params)
        self.type = "rooftop_solar"
        self.region = self.config["region"]
        self.state = self.config["state"]
        self.installation_cost_usd = self.config["installation_cost_usd"]

        # If yearly time series of tco2 sequestered is provided, use that as basis for project and project length
        self.kwh_per_yr = self.config["kwh_per_yr"]
        self.tco2_per_kwh = self.config["tco2_per_kwh"]
        self.usd_per_kwh = self.config["usd_per_kwh"]
        self.total_length_project = self.config["total_length_project"]

        self.avg_discounted_unit_value_solar_usd_per_yr = (
            self._compute_discounted_annual_avg_revenue()
        )
        self.usd_per_yr = self._compute_usd_per_yr()
        self.tco2_per_yr, self.total_tco2 = self._compute_tco2()
        self.present_value_of_installation_cost = (
            self._compute_discounted_annual_avg_cost()
        )
        self.cost_usd_per_tco2 = self._compute_unit_value_per_tco2(
            self.present_value_of_installation_cost
        )
        self.avg_discounted_unit_value_solar_usd_per_tco2 = (
            self._compute_unit_value_per_tco2(
                self.avg_discounted_unit_value_solar_usd_per_yr
            )
        )

        # once we've assigned all variables in the config file, delete the attribute
        delattr(self, "config")

    def _compute_discounted_annual_avg_revenue(self) -> float:
        """compute annual average discounted revenue of solar energy produced"""

        # compute average value of solar production (per year)
        return self._avg_discounted_unit_cash_flow(
            self.kwh_per_yr,
            [self.usd_per_kwh] * self.total_length_project,
        )

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
        total_tco2 = tco2_per_year * self.total_length_project

        return (tco2_per_year, total_tco2)

    def _compute_discounted_annual_avg_cost(self) -> float:
        """compute annual-average discounted cost of solar array"""
        # TODO eliminate hard coding for spending (currently all spending is in year 1)

        return self._avg_discounted_unit_cash_flow(
            self.installation_cost_usd,
            [1] + [0] * (self.total_length_project - 1),
        )

    def _compute_price_of_tco2_removal(self) -> float:
        """Computes the total cost of tco2 removal"""
        return self.installation_cost_usd / self.total_tco2

    def _compute_unit_value_per_tco2(self, value_usd_per_yr) -> float:
        return value_usd_per_yr / self.tco2_per_yr
