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
        self.discount_rate_real = (
            (1 + self.discount_rate) / (1 + self.inflation_rate)
        ) - 1
        if "discount_rate_real" in self.config:
            self.discount_rate_real = self.config["discount_rate_real"]

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

    def pv(self, rate, data_stream: List[float]) -> float:
        """computes the discounted present value of a data (cash?) stream
        Args:
            data_stream: stream to be discounted at the internal attribute self.discount_rate
        Returns:
            float that is the present value of the future stream
        """
        return npf.npv(rate, data_stream)

    def unit_conversion(
        self,
        data_start_unit: Union[List[float], float],
        conversion_factor: Union[List[float], float],
    ) -> Union[List[float], float]:
        """Helper method for converting values between units
        Args:
            data_start_unit: a list of floats or a float containing the data in the first unit
            conversion_factor: a list of floats or a float containing conversion factor(s) for data
        Returns:
            A list of floats or float with the original data converted to the new units
        """
        # TODO add error checking for lack of floats, either standalone or w/i list
        if isinstance(data_start_unit, list) and not isinstance(
            conversion_factor, list
        ):
            return [d * conversion_factor for d in data_start_unit]
        if isinstance(data_start_unit, list) and isinstance(conversion_factor, list):
            if len(data_start_unit) != len(conversion_factor):
                raise ValueError(
                    "If data to be converted and conversion factor are both lists, they must be the same length"
                )
            return [d * c for d, c in zip(data_start_unit, conversion_factor)]
        if not isinstance(data_start_unit, list) and isinstance(
            conversion_factor, list
        ):
            # TODO raise warning ... or maybe error? can conversion_factor be a list when data_start_unit isn't?
            return [data_start_unit * c for c in conversion_factor]
        return data_start_unit * conversion_factor

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
