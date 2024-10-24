""" Abstract base class for energy system simulation"""

# pylint: disable=too-many-instance-attributes
import logging
from abc import ABC, abstractmethod
from pathlib import PosixPath
from typing import Union

import numpy as np

from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)


class EnergySystem(ABC):
    """Parent class for energy system"""

    subclasses = {}

    @classmethod
    def register_subclass(cls, system_type: str):
        """creates decorator that automatically registers subclasses
        To use, decorate the child class definition with @Facility.register_subclass("[name-of-child-class]")
        """

        def decorator(subclass):
            cls.subclasses[system_type] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, system_type: str, config: Union[dict, Union[str, PosixPath]]):
        """Creates a new child class using the system_type (e.g., "logit")"""
        if system_type not in cls.subclasses:
            raise ValueError(f"Bad system_type {system_type}")
        if isinstance(config, dict):
            return cls.subclasses[system_type](config)
        return cls.subclasses[system_type](yaml_to_dict(config))

    def __init__(self, config: dict):
        # constants
        self.years = config["years"]
        self.oil_mwh_per_bbl = config["oil_mwh_per_bbl"]
        self.eor_recovery_factor_bbl_per_tco2 = config["recovery_factor"]
        self.sources = config["sources"]  # Gc
        self.reservoirs = config["reservoirs"]
        self.carbon_intensity_kg_per_mwh = config["carbon_intensity_kg_per_mwh"]
        self.ccs_prevalence_frac = config["ccs_prevalence"]
        self.eor_frac = config["eor_frac"]
        self.energy_penalty = config["energy_penalty"]
        self.ccs_capture_frac = config["ccs_capture_frac"]

        # variables
        self.timestep = 0
        self.stop_timestep = len(self.years)
        self.energy_shares = self._initialize_timeseries(config["energy_shares"])
        self.energy_costs_usd_per_mwh = self._initialize_timeseries(
            config["energy_costs_usd_per_mwh"]
        )
        self.carbon_to_atmosphere_kg_per_mwh = self._initialize_timeseries(
            dict(zip(self.sources, [0] * len(self.sources)))
        )
        self.carbon_captured_kg_per_mwh = self._initialize_timeseries(
            dict(zip(self.sources, [0] * len(self.sources)))
        )
        self.reservoir_balances = self._initialize_timeseries(
            dict(zip(self.reservoirs, [0] * len(self.reservoirs)))
        )

    def _initialize_timeseries(self, starting_dict):
        """initialize an object to hold simulation outputs"""
        ts_dict = {}
        for k, v in starting_dict.items():
            ts_dict[k] = [v]
        ts_dict["years"] = self.years
        return ts_dict

    def _adjust_price(self, price_per_one_unit, change_in_units):
        return price_per_one_unit / (1 + change_in_units)

    def simulate(self):
        """Simulate energy system for all timesteps"""
        while self.timestep < self.stop_timestep:
            self.move_carbon()
            self.update_energy_share()
            self.update_prices()
            self.timestep += 1

    @abstractmethod
    def update_prices(self):
        """Adjust energy prices based on various system characteristics (e.g., subsidies)"""

    @abstractmethod
    def update_energy_share(self):
        """Use method to update share of primary energy provided by each source"""

    @abstractmethod
    def move_carbon(self):
        """Shift carbon between reservoirs"""


@EnergySystem.register_subclass("logit")
class LogitEnergySystem(EnergySystem):
    """Energy system class that uses a logit function to update shares of primary energy by source"""

    def __init__(self, config):
        # initialize parent class __init__
        super().__init__(config)
        self.system_type = "logit"
        self.logit_exponent = config["logit_exponent"]
        # TODO  error checking

    def update_energy_share(self):
        """Update share w/ logit function: balance b/w cost & existing share controlled by logit exponent"""
        current_share = []
        costs = []
        # get current shares for each source as well as costs, raised to the logit exponent
        for s in self.sources:
            current_share.append(self.energy_shares[s][self.timestep])
            costs.append(
                (self.energy_costs_usd_per_mwh[s][self.timestep]) ** self.logit_exponent
            )

        new_share_denominator = np.sum(
            [share * cost for share, cost in zip(current_share, costs)]
        )
        for s, share, cost in zip(self.sources, current_share, costs):
            self.energy_shares[s][self.timestep + 1] = (
                share * cost
            ) / new_share_denominator

    def move_carbon(self):
        """Computes carbon flow per unit energy, from each source, to the atmosphere and to CCS capture"""
        to_atmosphere = 0
        captured = 0
        for s in self.sources:
            # compute carbon that is captured by source (units of kg per mwh of energy)
            self.carbon_captured_kg_per_mwh[s][self.timestep] = (
                self.ccs_prevalence_frac[self.timestep]
                * self.ccs_capture_frac[s]
                * self.energy_shares[s][self.timestep]
                * self.carbon_intensity_kg_per_mwh[s]
            )
            # compute carbon emitted to the atmosphere by source (units of kg per mwh of energy)
            self.carbon_to_atmosphere_kg_per_mwh[s][self.timestep] = (
                self.energy_shares[s][self.timestep]
                * self.carbon_intensity_kg_per_mwh[s]
                - self.carbon_captured_kg_per_mwh[s][self.timestep]
            )

            to_atmosphere = (
                to_atmosphere + self.carbon_to_atmosphere_kg_per_mwh[s][self.timestep]
            )
            captured = captured + self.carbon_captured_kg_per_mwh[s][self.timestep]

        # compute/record updated balances of carbon in reservoirs
        prev_timestep = self.timestep - 1
        if self.timestep == 0:
            prev_timestep = 0

        # TODO refactor to find cleaner way to track this
        self.reservoir_balances["atmosphere"][self.timestep] = (
            self.reservoir_balances["atmosphere"][prev_timestep] + to_atmosphere
        )
        self.reservoir_balances["ccs"][self.timestep] = (
            self.reservoir_balances["ccs"][prev_timestep] + captured
        )

    def update_prices(self):
        """Adjusts prices based on shifts of CO2"""
        current_oil_price = self.energy_costs_usd_per_mwh["oil"][self.timestep]
        # compute the number of mwh of energy produced with eor
        mwh_oil_produced_with_eor = (
            self.reservoir_balances["ccs"][self.timestep]
            * self.eor_frac[self.timestep]
            * self.eor_recovery_factor_bbl_per_tco2
            * self.oil_mwh_per_bbl
            / 1000
        )  # 1000 kg in one ton
        # compute revised price for next timestep
        self.energy_costs_usd_per_mwh["oil"][self.timestep + 1] = self._adjust_price(
            current_oil_price, mwh_oil_produced_with_eor
        )
        # gas is more expensive because of CCS in ng processing
        current_ng_price = self.energy_costs_usd_per_mwh["ng"][self.timestep]
        self.energy_costs_usd_per_mwh["ng"][self.timestep + 1] = self._adjust_price(
            current_ng_price, self.energy_penalty["ng"]
        )
