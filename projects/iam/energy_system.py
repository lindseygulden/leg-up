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

    def __init__(self, config_info: Union[str, dict]):
        if isinstance(config_info, str):
            config = yaml_to_dict(config_info)
        else:
            config = config_info
        # constants
        self.kg_per_ton = 1000
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
        self.stop_timestep = len(self.years)
        self.base_cost_energy_usd_per_mwh = config["base_cost_energy_usd_per_mwh"]
        self.sec_45q_usd_tco2 = config["sec_45q_usd_tco2"]

        # variables
        self.timestep = 0
        self.ccs_kgco2_per_mwh = {self.timestep: {"eor": 0, "gs": 0}}
        self.ccs_usd_per_mwh = {self.timestep: {"eor": 0, "gs": 0}}
        self.energy_shares = {self.timestep: config["energy_shares"]}
        self.adjusted_cost_energy_usd_per_mwh = {
            self.timestep: config["base_cost_energy_usd_per_mwh"]
        }
        self.carbon_flow_kg_per_mwh = {}
        for r in self.reservoirs:
            self.carbon_flow_kg_per_mwh[r] = {
                self.timestep: dict(zip(self.sources, [0] * len(self.sources)))
            }

    def _assemble_df(self):
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
            logging.info(
                " >>> Simulating %s (timestep %s)",
                self.years[self.timestep],
                self.timestep,
            )
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

    def __init__(self, config_info):
        # initialize parent class __init__
        super().__init__(config_info)
        if isinstance(config_info, str):
            config = yaml_to_dict(config_info)
        else:
            config = config_info
        self.system_type = "logit"
        self.logit_exponent = config["logit_exponent"]
        # TODO  error checking

    def move_carbon(self):
        """Computes carbon flow per unit energy, from each source, to the atmosphere and to CCS capture"""

        to_atmosphere_dict = {}
        captured_dict = {}
        for s in self.sources:
            # compute carbon that is captured by source (units of kg per mwh of energy)
            captured_dict[s] = (
                self.ccs_prevalence_frac[self.timestep]
                * self.ccs_capture_frac[s]
                * self.energy_shares[self.timestep][s]
                * self.carbon_intensity_kg_per_mwh[s]
            )

            # compute carbon emitted to the atmosphere by source (units of kg per mwh of energy)
            to_atmosphere_dict[s] = (
                self.energy_shares[self.timestep][s]
                * self.carbon_intensity_kg_per_mwh[s]
                - captured_dict[s]
            )

        self.carbon_flow_kg_per_mwh["ccs"][self.timestep] = captured_dict
        self.carbon_flow_kg_per_mwh["atmosphere"][self.timestep] = to_atmosphere_dict

        total_kgco2_injected_with_ccs = np.sum(
            list(self.carbon_flow_kg_per_mwh["ccs"][self.timestep].values())
        )
        # quantify total kg co2 which ccs's co2 is injected for eor and for geologic storage

        self.ccs_kgco2_per_mwh = self.ccs_kgco2_per_mwh | {
            self.timestep
            + 1: {
                "eor": total_kgco2_injected_with_ccs * self.eor_frac[self.timestep],
                "gs": total_kgco2_injected_with_ccs
                * (1 - self.eor_frac[self.timestep]),
            }
        }

        self.ccs_usd_per_mwh = self.ccs_usd_per_mwh | {
            self.timestep
            + 1: {
                "eor": self.ccs_kgco2_per_mwh[self.timestep + 1]["eor"]
                * self.sec_45q_usd_tco2["eor"]
                / self.kg_per_ton,
                "gs": self.ccs_kgco2_per_mwh[self.timestep + 1]["gs"]
                * self.sec_45q_usd_tco2["gs"]
                / self.kg_per_ton,
            }
        }

    def update_energy_share(self):
        """Update share w/ logit function: balance b/w cost & existing share controlled by logit exponent"""
        current_share = []
        new_share_dict = {}
        costs = []

        # get current shares for each source as well as costs, raised to the logit exponent
        for s in self.sources:
            current_share.append(self.energy_shares[self.timestep][s])
            costs.append(
                (self.adjusted_cost_energy_usd_per_mwh[self.timestep][s])
                ** self.logit_exponent
            )

        new_share_denominator = np.sum(
            [share * cost for share, cost in zip(current_share, costs)]
        )
        for s, share, cost in zip(self.sources, current_share, costs):
            new_share_dict[s] = (share * cost) / new_share_denominator

        self.energy_shares[self.timestep + 1] = new_share_dict

    def update_prices(self):
        """Adjusts prices for next time step based on shifts of CO2"""
        # start by making next time step's prices the same as this time step
        self.adjusted_cost_energy_usd_per_mwh[self.timestep + 1] = (
            self.base_cost_energy_usd_per_mwh.copy()
        )

        # adjust oil price for eor
        # base_oil_price = self.base_cost_energy_usd_per_mwh[self.timestep]["oil"]
        base_oil_price = self.base_cost_energy_usd_per_mwh["oil"]

        # compute the number of mwh of energy produced with eor
        mwh_oil_produced_with_eor = (
            np.sum(list(self.carbon_flow_kg_per_mwh["ccs"][self.timestep].values()))
            * self.eor_frac[self.timestep]
            * self.eor_recovery_factor_bbl_per_tco2
            * self.oil_mwh_per_bbl
            / self.kg_per_ton
        )  # 1000 kg in one ton
        # compute revised price for next timestep
        self.adjusted_cost_energy_usd_per_mwh[self.timestep + 1]["oil"] = (
            self._adjust_price(base_oil_price, mwh_oil_produced_with_eor)
            - self.ccs_usd_per_mwh[self.timestep]["eor"]
        )

        # gas is more expensive because of CCS in ng processing
        # base_ng_price = self.base_cost_energy_usd_per_mwh[self.timestep]["ng"]
        base_ng_price = self.base_cost_energy_usd_per_mwh["ng"]
        self.adjusted_cost_energy_usd_per_mwh[self.timestep + 1]["ng"] = (
            self._adjust_price(
                base_ng_price,
                (
                    -1
                    * self.ccs_prevalence_frac[self.timestep]
                    * self.energy_penalty["ng"]
                    * self.energy_shares[self.timestep]["ng"]
                ),
            )
        )
