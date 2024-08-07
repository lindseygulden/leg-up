"""Builds subclass for project that represents a rooftop solar project"""

# pylint: disable=too-many-instance-attributes,too-few-public-methods
import logging
from pathlib import PosixPath
from typing import Tuple, Union

import numpy as np
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
        # TODO switch to pycharm objects
        self.tco2_per_kwh = self.config["tco2_per_kwh"]  # can be list or float
        self.usd_per_kwh = self.config["usd_per_kwh"]  # can be list or float

        # financing details
        self.bond_interest_rate = 0.05
        if "bond_interest_rate" in self.config:
            self.bond_interest_rate = self.config["bond_interest_rate"]
        # get dollars and tco2 per year given the specs of the solar system
        self.usd_per_yr, self.tco2_per_yr = self._convert_solar_units()

        self.total_tco2 = np.sum(self.tco2_per_yr)

        # compute financials
        # Compute value-to-consumer of electricity generated by pv array
        self.pv_solar_revenue_usd = self.pv(self.discount_rate_real, self.usd_per_yr)
        # Compute expenses -- currently just install cost...but could be others?

        # total amount due for a loan
        self.bond_payments_usd = self._compute_municipal_bond_payments()

        # homeowner portion (for effective zero-interest loan, funded by govt)
        self.homeowner_portion_bond_payment_usd = [
            self.installation_cost_usd / self.project_length_yrs
        ] * self.project_length_yrs

        # government portion (paying interest on the bond, to enable zero-interest)
        self.govt_portion_bond_payment_usd = [
            b - h
            for b, h in zip(
                self.bond_payments_usd, self.homeowner_portion_bond_payment_usd
            )
        ]
        self.pv_total_expense_usd = self.pv(
            self.discount_rate_real,
            self.bond_payments_usd,
        )
        self.pv_homeowner_expense_usd = self.pv(
            self.discount_rate_real,
            self.homeowner_portion_bond_payment_usd,
        )

        self.pv_govt_expense_usd = self.pv(
            self.discount_rate_real, self.govt_portion_bond_payment_usd
        )

        self.npv = self.pv_solar_revenue_usd - self.pv_total_expense_usd
        self.npv_homeowner_usd = (
            self.pv_solar_revenue_usd - self.pv_homeowner_expense_usd
        )

        # express present value in terms of units of co2 and per year
        self.pv_total_expense_usd_per_tco2 = self.pv_total_expense_usd / self.total_tco2
        self.pv_homeowner_expense_usd_per_tco2 = (
            self.pv_homeowner_expense_usd / self.total_tco2
        )
        self.pv_govt_expense_usd_per_tco2 = self.pv_govt_expense_usd / self.total_tco2
        self.pv_solar_usd_per_tco2 = self.pv_solar_revenue_usd / self.total_tco2
        self.npv_homeowner_usd_per_tco2 = self.npv_homeowner_usd / self.total_tco2

        # once we've assigned all variables in the config file, delete the attribute
        delattr(self, "config")

    def _convert_solar_units(self) -> Tuple[float, float]:
        """Converts kwh for time period to usd equivalent and equivalent tco2 averted"""
        usd_per_yr = self.unit_conversion(self.kwh_per_yr, self.usd_per_kwh)
        tco2_per_yr = self.unit_conversion(self.kwh_per_yr, self.tco2_per_kwh)
        # if usd_per_yr and/or tco2_per_yr are floats (not lists), replace w/ lists of same length as n. proj. yrs
        if isinstance(usd_per_yr, float):
            usd_per_yr = [usd_per_yr] * self.project_length_yrs

        if isinstance(tco2_per_yr, float):
            tco2_per_yr = [tco2_per_yr] * self.project_length_yrs

        return usd_per_yr, tco2_per_yr

    def _compute_municipal_bond_payments(self):
        """Assume equal principal + interest payments for each year for a loan term of project_length_yr"""
        return [
            -1
            * npf.pmt(
                self.bond_interest_rate,
                self.project_length_yrs,
                self.installation_cost_usd,
                0,
            )
        ] * self.project_length_yrs
