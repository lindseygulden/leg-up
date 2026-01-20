"""Energy system model class (abstract base class + variations) to simulate energy-source shares"""

# pylint: disable=too-many-positional-arguments,too-many-instance-attributes
import logging
from abc import ABC, abstractmethod
from pathlib import PosixPath
from typing import Union

import numpy as np
import pandas as pd

from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)


class IAM(ABC):
    """Parent class for miniature representation of energy system"""

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
        """general class initiaton from either configuration yaml path or from dictionary"""
        if isinstance(config_info, str):
            config = yaml_to_dict(config_info)
        else:
            config = config_info
        # constants
        self.start_yr = config["start_yr"]
        self.timestep_yr = config["timestep_yr"]
        self.n_steps = config["n_steps"]

        self.energy_demand_growth_rate_per_timestep = config[
            "energy_demand_growth_rate_per_timestep"
        ]

        # get price data
        if "usd_per_tco2" in config:
            self.usd_per_tco2 = config["usd_per_tco2"]
        elif "price_curve" in config:
            # assigns value to self.usd_per_tco2
            self._compute_carbon_price_curve(config["price_curve"])
        else:
            logging.info(
                "No CO2 price information provided in config. Assuming price is $0/tCO2"
            )
            self.usd_per_tco2 = [0] * self.n_steps

        self.years = [
            int(self.start_yr + s * self.timestep_yr) for s in range(self.n_steps + 1)
        ]
        self.energy_sources = config["energy_sources"]
        self.df = pd.DataFrame.from_dict(
            config["parameters"]
        )  # format is multilayered dict with [non-time-varying parameter names][energy source name]

        self.step_count = 0
        self.shares = {0: self.df["starting_share"].to_dict()}

        self.retirement_share = {}
        self.frac_for_allocation = {}
        self.share_of_new = {}

        # self.adjust_co2_emission_rates()
        self.compute_price_of_energy_generation()
        self.compute_base_price_of_cdr()
        self.compute_price_of_net_carbon_emissions()
        self.compute_adjusted_prices()

    def _sigmoid_curve(self, x, ub, lb, inflection, steepness):
        return ub + (lb - ub) / (1 + (x / inflection) ** steepness)

    def _compute_carbon_price_curve(self, price_curve):
        """computes a price curve from input parameters"""

        # get integer-valued timesteps
        times = list(range(self.n_steps + 1))

        # compute prices
        if price_curve["type"] == "sigmmoid":
            self.usd_per_tco2 = [
                self._sigmoid_curve(
                    t,
                    price_curve["upper_bound"],
                    price_curve["lower_bound"],
                    price_curve["inflection"] * self.n_steps,
                    price_curve["steepness"],
                )
                for t in times
            ]
        elif price_curve["type"] == "line":
            self.usd_per_tco2 = [
                price_curve["slope"] * t + price_curve["intercept"] for t in times
            ]
        elif price_curve["type"] == "minmax":
            increment = (
                price_curve["total_increase_in_price"] - price_curve["starting_price"]
            ) / self.n_steps
            self.usd_per_tco2 = [
                price_curve["starting_price"] + increment * t
                for t in range(self.n_steps + 1)
            ]

    def compute_base_price_of_cdr(self):
        """computes base price to implementcarbon removal for each timestep"""
        self.price_of_cdr_usd_per_mwh = {
            ts: dict(
                zip(
                    self.energy_sources,
                    [
                        self.df.loc[e, "starting_carbon_removal_price_fraction"]
                        * self.df.loc[e, "starting_energy_generation_price_usd_per_mwh"]
                        * (1 - self.df.loc[e, "frac_cdr_cost_decrease_per_timestep"])
                        ** ts
                        for e in self.energy_sources
                    ],
                )
            )
            for ts in range(self.n_steps + 1)
        }

    def compute_price_of_energy_generation(self):
        """adjusts prices for energy for each time step using the starting price and the learning curve"""
        self.price_of_energy_generation_usd_per_mwh = {
            ts: dict(
                zip(
                    self.energy_sources,
                    [
                        self.df.loc[e, "starting_energy_generation_price_usd_per_mwh"]
                        * (
                            1
                            - self.df.loc[
                                e, "frac_energy_generation_cost_decrease_per_timestep"
                            ]
                        )
                        ** ts
                        for e in self.energy_sources
                    ],
                )
            )
            for ts in range(self.n_steps + 1)
        }

    def compute_price_of_net_carbon_emissions(self):
        """Computes the contribution to energy-source price from CDR CO2 removals and missed CO2 emissions"""
        self.net_price_of_carbon_emissions_usd_per_mwh = {
            ts: dict(
                zip(
                    self.energy_sources,
                    [
                        self.price_of_cdr_usd_per_mwh[ts][e]
                        + self.usd_per_tco2[ts]
                        * self.df.loc[e, "co2_per_mwh"]
                        * (1 - self.df.loc[e, "capture_fraction"])
                        for e in self.energy_sources
                    ],
                )
            )
            for ts in range(self.n_steps + 1)
        }

    def compute_adjusted_prices(self):
        """Finds total price per MWh for each energy source,
        summing cost of electricity generation and net cost of carbon emissions"""

        self.usd_per_mwh = {
            ts: dict(
                zip(
                    self.energy_sources,
                    [
                        self.price_of_energy_generation_usd_per_mwh[ts][e]
                        + self.net_price_of_carbon_emissions_usd_per_mwh[ts][e]
                        for e in self.energy_sources
                    ],
                )
            )
            for ts in range(self.n_steps + 1)
        }

    def compute_retirement_fraction(self):
        """compute the fraction of total energy generation that is retired at end of project life"""
        # temporary variables for ease of reading code below
        current_shares = [self.shares[self.step_count][e] for e in self.energy_sources]
        lifespans = [self.df.loc[e, "lifespan_yr"] for e in self.energy_sources]
        retire_yet = [
            1 if self.step_count >= self.df.loc[e, "retire_timestep"] else 0
            for e in self.energy_sources
        ]
        self.retirement_share[self.step_count] = dict(
            zip(
                self.energy_sources,
                [
                    r * s * y
                    for r, s, y in zip(
                        [self.timestep_yr / l for l in lifespans],
                        current_shares,
                        retire_yet,
                    )
                ],
            )
        )

    def update_shares(self):
        """update the shares of energy according to most recent calculations"""
        # temporary variables for ease of reading code below

        current_shares = [self.shares[self.step_count][e] for e in self.energy_sources]
        share_of_new = [
            self.share_of_new[self.step_count][e] for e in self.energy_sources
        ]
        retirement_share = [
            self.retirement_share[self.step_count][e] for e in self.energy_sources
        ]
        remaining_shares = [
            (c - r) / (1 + self.energy_demand_growth_rate_per_timestep)
            for c, r in zip(current_shares, retirement_share)
        ]
        self.frac_for_allocation[self.step_count] = 1.0 - np.sum(remaining_shares)

        # update share fractions based on fraction of facilities retiring, new share allocaiton, current shares
        alpha_share_new = [
            r + self.frac_for_allocation[self.step_count] * s
            for r, s in zip(remaining_shares, share_of_new)
        ]
        # update the timestep
        self.step_count += 1
        self.shares[self.step_count] = dict(zip(self.energy_sources, alpha_share_new))

    def simulate(self, return_data=False):
        """simulate the shares of technology using parameters given"""

        for _ in range(self.n_steps):
            self.compute_new_shares()

            self.compute_retirement_fraction()

            self.update_shares()

        if return_data is True:
            return pd.DataFrame(self.shares)
        return None

    def report(self):
        """Placeholder for a reporting function"""
        print(self.usd_per_mwh)

    @abstractmethod
    def compute_new_shares(self):
        """abstract method that can change how shares are allocated (e.g., modified logit)"""


@IAM.register_subclass("nestedlogit")
class NestedLogitIAM(IAM):
    """Energy system class that uses a logit function to update shares of primary energy by source"""

    def __init__(self, config_info):
        # initialize parent class __init__
        super().__init__(config_info)

        if isinstance(config_info, str):
            config = yaml_to_dict(config_info)
        else:
            config = config_info

        self.system_type = "nestedlogit"
        self.logit_exponents = config[
            "logit_exponents"
        ]  # nested list of sectors and exponents
        # TODO  error checking

    def _modified_logit_shares(self, current_shares, current_prices, logit_exponent):
        """compute modified logit share according to GCAM documenation
        n.b.: see: https://jgcri.github.io/gcam-doc/choice.html"""
        denominator = np.dot(
            current_shares,
            [p**logit_exponent for p in current_prices],
        )

        return [
            (a * p**logit_exponent) / denominator
            for a, p in zip(current_shares, current_prices)
        ]

    def compute_new_shares(self):
        """nested modified logit for computing new shares"""
        current_shares = self.shares[self.step_count]
        if self.step_count == 0:
            current_shares = self.df["starting_share"].to_dict()
        calc_df = pd.DataFrame(
            [
                self.df["subsector"].to_dict(),
                current_shares,
                self.usd_per_mwh[self.step_count],
            ],
            index=["subsector", "source_share_of_total", "price"],
        ).T

        calc_df.index.name = "energy_source"

        calc_subsec_totals_df = (
            calc_df[["subsector", "source_share_of_total"]]
            .groupby("subsector")
            .sum()
            .reset_index()
        )

        calc_subsec_totals_df.columns = [
            "subsector",
            "subsector_share_of_total",
        ]

        calc_df = calc_df.reset_index().merge(
            calc_subsec_totals_df[
                [
                    "subsector",
                    "subsector_share_of_total",
                ]
            ],
            on="subsector",
        )
        calc_df["share_of_subsector"] = (
            calc_df["source_share_of_total"] / calc_df["subsector_share_of_total"]
        )

        # compute subsector shares
        share_of_subsector_allocatable_dict = {}

        subset_list = []
        for subsector in self.logit_exponents["subsector"].keys():
            calc_subset_df = calc_df.loc[calc_df.subsector == subsector].set_index(
                "energy_source"
            )
            calc_subset_df[
                "share_of_subsector_allocatable"
            ] = self._modified_logit_shares(
                calc_subset_df["share_of_subsector"],
                calc_subset_df["price"],
                self.logit_exponents["subsector"][subsector],
            )
            subset_list.append(calc_subset_df)

            share_of_subsector_allocatable_dict = (
                share_of_subsector_allocatable_dict
                | (calc_subset_df["share_of_subsector_allocatable"].to_dict())
            )

        calc_df = pd.concat(subset_list)

        calc_df["share_times_price"] = (
            calc_df["share_of_subsector_allocatable"] * calc_df["price"]
        )
        subsector_avg_price_dict = (
            calc_df[["share_times_price", "subsector"]]
            .groupby("subsector")
            .sum()
            .to_dict()["share_times_price"]
        )

        calc_df["subsector_avg_price"] = [
            subsector_avg_price_dict[s] for s in calc_df["subsector"]
        ]

        calc_grouped_subset_df = (
            calc_df.groupby("subsector")
            .first()[["subsector_avg_price", "subsector_share_of_total"]]
            .copy()
        )

        # compute the shares across top-level sectors
        grouped_dict = {}
        for sector in self.logit_exponents["sector"].keys():
            calc_grouped_subset_df["new_sector_share"] = self._modified_logit_shares(
                calc_grouped_subset_df["subsector_share_of_total"],
                calc_grouped_subset_df["subsector_avg_price"],
                self.logit_exponents["sector"][sector],
            )
            grouped_dict = (
                grouped_dict | calc_grouped_subset_df["new_sector_share"].to_dict()
            )

        calc_df["new_sector_share"] = [grouped_dict[s] for s in calc_df.subsector]

        calc_df = pd.merge(
            calc_df,
            pd.DataFrame(
                [share_of_subsector_allocatable_dict],
                index=["new_source_share_of_total"],
            ).T,
            left_index=True,
            right_index=True,
        )
        calc_df["new_share"] = (
            calc_df["new_sector_share"] * calc_df["new_source_share_of_total"]
        )
        self.share_of_new[self.step_count] = calc_df["new_share"].to_dict()


# TODO finish this subclass
@IAM.register_subclass("logit")
class LogitIAM(IAM):
    """Energy system class that uses a logit function to update shares of primary energy by source"""

    def __init__(self, config_info):
        # initialize parent class __init__
        super().__init__(config_info)
        if isinstance(config_info, str):
            config = yaml_to_dict(config_info)
        else:
            config = config_info
        self.system_type = "logit"
        self.logit_exponent = config["logit_exponent"]  # single exponent for everything
        # TODO  error checking


def compute_new_shares(self):
    """modified logit for computing new shares"""
    # temporary variables for ease of reading code below
    current_shares = [self.shares[self.step_count][e] for e in self.energy_sources]
    current_prices = [self.usd_per_mwh[self.step_count][e] for e in self.energy_sources]

    denom = np.dot(current_shares, [p**self.logit_exponent for p in current_prices])

    self.share_of_new[self.step_count] = dict(
        zip(
            self.energy_sources,
            [
                (a * p**self.logit_exponent) / denom
                for a, p in zip(current_shares, current_prices)
            ],
        )
    )
