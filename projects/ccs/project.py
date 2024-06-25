""" Abstract base class for unit-economics project-economics simulation """

import logging
from abc import ABC
from pathlib import PosixPath
from typing import List, Union

import numpy_financial as npf

from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)


class Project(ABC):
    """Parent class for various project simulations connected to CCS evaluations"""

    def __init__(self, params: Union[Union[str, PosixPath], dict]):
        if isinstance(params, dict):
            self.config = params
        else:
            self.config = yaml_to_dict(params)

        self.project_length_yrs = self.config["project_length_yrs"]
        self.inflation_rate = self.config["inflation_rate"]
        self.discount_rate = self.config["discount_rate"]

        self.project_name = "project"
        if "project_name" in self.config:
            self.project_name = self.config["project_name"]

        # location
        self.lat = 40.0
        self.lon = -90.0
        if "lat" in self.config:
            self.lat = self.config["lat"]
        if "lon" in self.config:
            self.lon = self.config["lon"]

    def describe(self):
        """describes data in the project"""
        return_str = "{self.project_n}:\n"
        for k, v in self.__dict__.items():
            return_str = return_str + (f"   - {k}: {v}\n")
        return return_str

    def avg_npv(self, cash_stream: List[float]):
        """computes the time-period avg. npv of a cash stream
        Args:
            discount_rate: rate (between [0,1]) of discount for future values
            fv: each value is the future value of a cash flow stream for a given time period period
        Returns:
            float that is the time-period-averaged net present value of the future cash stream
        """
        return npf.npv(self.discount_rate, cash_stream) / len(cash_stream)

    def inflate(self, revenue_stream: List[float]):
        """computes impact of inflation on a revenue stream expressed in today's dollars"""
        if isinstance(self.inflation_rate, list):
            # TODO add error checking to check for same-length lists and/or handle different lengths
            inflated_revenue_stream = [
                -1 * npf.fv(self.inflation_rate, i + 1, 0, x)
                for i, x in zip(self.inflation_rate, revenue_stream)
            ]
        elif isinstance(self.inflation_rate, float):  # assumed constant inflation rate
            inflated_revenue_stream = [
                -1 * npf.fv(self.inflation_rate, i + 1, 0, x)
                for i, x in enumerate(revenue_stream)
            ]
        else:
            raise TypeError("Inflation rate must be either a float or a list of floats")
        return inflated_revenue_stream

    def _avg_discounted_unit_cash_flow(self, unit_per_period, price_per_unit):
        """computes time-period-average discounted unit cash flow
        Args:
            unit_per_period: list of floats, specifying amount of item sold/bought per period
            price_per_unit: list of floats, specifying amount of cash exchanged per unit
        Returns:
            the time-averaged present value of the inflation-adjusted cash flow
        """
        if len(unit_per_period) != len(price_per_unit):
            raise ValueError(
                "Arguments specifying units exchanged and price per unit must be equal-lengthed lists"
            )
        cash_flow = [b * p for b, p in zip(unit_per_period, price_per_unit)]
        inflated_cash_flow = self.inflate(cash_flow)
        return self.avg_npv(inflated_cash_flow)
