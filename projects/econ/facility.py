""" Abstract base class for facility """

import logging
from abc import ABC,abstractmethod
from pathlib import PosixPath
from typing import List, Union

from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)


class Facility(ABC):
    """Parent class for CCS facility operations"""

        """Parent class for various thermodynamic funcitons describing energy needed for CO2 separation"""

    subclasses = {}

    @classmethod
    def register_subclass(cls, facility_type: str):
        """creates decorator that automatically registers subclasses
        To use, decorate the child class definition with @Facility.register_subclass("[name-of-child-class]")
        """

        def decorator(subclass):
            cls.subclasses[facility_type] = subclass
            return subclass

        return decorator

    @classmethod
    def create(
        cls, facility_type: str, config: Union[dict,Union[str,PosixPath]]
    ):
        """Creates a new child class using the facility_type (e.g., "ng_power_plant")"""
        if facility_type not in cls.subclasses:
            raise ValueError(f"Bad facility_type {facility_type}")
        if isinstance(config,dict):
            return cls.subclasses[facility_type](config)
        return cls.subclasses[facility_type](yaml_to_dict(config))

    def __init__(self, config:dict):

        # constants
        self.carbon_intensity_oil_tco2_per_bbl = config["carbon_intensity_oil_tco2_per_bbl"] # Gc


        # variables
        self.frac_co2_captured = config["frac_co2_captured"] # α
        self.frac_captured_co2_used_for_eor = config["frac_captured_co2_used_for_eor"] # 𝛽


        self.q45_subsidy_usd_per_tco2 = config["q45_subsidy_usd_per_tco2"] # (Qseq and Qeor)

        self.eor_recovery_factor_bbl_per_tco2 = config["eor_recovery_factor_bbl_per_tco2"] # Θ

        self.oil_price_usd_per_bbl = config["oil_price_usd_per_bbl"] # Cc

        self.energy_function_mwh_per_tco2=EnergyFunction.create(self.config['energy_function'],self.config['energy_function_parameters'])  # f(α)

    @abstractmethod
    def compute_gross_profit(self):
        pass

    @abstractmethod
    def compute_emissions(self):
        pass

@Facility.register_subclass("ng_power_plant")
class NGPowerPlantFacility(Facility):
    def __init__(self, config):
        # initialize parent class __init__
        super().__init__(config)
        self.facility_type = "ng_power_plant"

        self.electricity_price_usd_per_mwh = config["electricity_price_usd_per_mwh"] # Pe
        self.gas_to_produce_electricity_tch4_per_mwh = config["gas_to_produce_electricity_tch4_per_mwh"] # Eg
        self.cost_of_natural_gas_usd_per_tch4 = config["cost_of_natural_gas_usd_per_tch4"] # Cg
        self.fraction_of_grid_power_produced_from_ng = config["fraction_of_grid_power_produced_from_ng"] # 𝜸
        self.carbon_intensity_natural_gas_tco2_per_tch4 = config["carbon_intensity_natural_gas_tco2_per_tch4"] # Gg

        attributes = self.__dict__
        if (self.electricity_price_usd_per_mwh <0) | (self.cost_of_natural_gas_usd_per_tch4 <0):
            raise ValueError(
                "Price of electricity and natural gas must be >= $0"
            )
        #TODO more error checking

    def _power_sales_profit(self):
        return self.electricity_price_usd_per_mwh - self.gas_to_produce_electricity_tch4_per_mwh*self.cost_of_natural_gas_usd_per_tch4

    def _ccs_profit(self,alpha):
        '''Computes the profit [USD/MWh] from the storage of carbon dioxide
        Args:
            alpha: fraction of emitted carbon dioxide that is injected for CCS
        Returns:
            ccs_net_profit_usd_per_mwh: amount profited (positve is income to operator) from subsidies
        '''
        cost_of_storage_usd_per_tco2 = self.energy_function_mwh_per_tco2.evaluate(alpha)*self.electricity_price_usd_per_mwh

        self.ccs_net_profit_usd_per_mwh=tco2_captured_per_mwh*(self.q45_subsidy_usd_per_tco2 - cost_of_storage_usd_per_tco2)


    def _eor_profit(self,alpha):

        self.eor_profit_usd_per_bbl alpha * self.eor_recovery_factor_bbl_per_tco2 * self.oil_price_usd_per_bbl
    def compute_gross_profit(self,fa):

        # intermediate variable for multiple calculations:
        self.tco2_captured_per_mwh =  alpha*self.gas_to_produce_electricity_tch4_per_mwh*self.carbon_intensity_natural_gas_tco2_per_tch4

        self.profit_from_power_sales_usd_per_mwh=self._power_sales_profit(self)
        self.profit_from_ccs_usd_per_mwh=self._ccs_profit(self)
        self.profit_from_eor_usd_per_mwh=self._eor_profit(self)
