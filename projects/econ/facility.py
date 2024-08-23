""" Abstract base class for facility """

import logging
from abc import ABC
from pathlib import PosixPath
from typing import List, Union

from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)


class Facility(ABC):
    """Parent class for CCS facility operations"""

    def __init__(self, params: Union[Union[str, PosixPath], dict]):
        if isinstance(params, dict):
            self.config = params
        else:
            self.config = yaml_to_dict(params)

        self.frac_co2_captured = self.config["frac_co2_captured"] # α
        self.frac_captured_co2_used_for_eor = self.config["frac_captured_co2_used_for_eor"] # 𝛽 
        
        self.electricity_price_usd_per_mwh = self.config["electricity_price_usd_per_mwh"] # Pe
        self.gas_to_produce_electricity_tch4_per_mwh = self.config["gas_to_produce_electricity_tch4_per_mwh"] # Eg
        self.cost_of_natural_gas_usd_per_tch4 = self.config["cost_of_natural_gas_usd_per_tch4"] # Cg
        self.q45_subsidy_usd_per_tco2 = self.config["q45_subsidy_usd_per_tco2"] # (Qseq and Qeor)
        self.carbon_intensity_natural_gas_tco2_per_tch4 = self.config["carbon_intensity_natural_gas_tco2_per_tch4"] # Gg
        self.eor_recovery_factor_bbl_per_tco2 = self.config["eor_recovery_factor_bbl_per_tco2"] # Θ
        self.fraction_of_grid_power_produced_from_ng = self.config["fraction_of_grid_power_produced_from_ng"] # 𝜸
        self.oil_price_usd_per_bbl = self.config["oil_price_usd_per_bbl"] # Cc
        self.carbon_intensity_oil_tco2_per_bbl = self.config["carbon_intensity_oil_tco2_per_bbl"] # Gc



        if "project_name" in self.config:
            self.project_name = self.config["project_name"]

        

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
