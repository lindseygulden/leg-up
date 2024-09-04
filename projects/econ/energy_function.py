""" Abstract base class for thermodynamic function defining energy needed to separate CO2 """

import logging
from abc import ABC, abstractmethod
from pathlib import PosixPath
from typing import List, Union, Dict, Optional

from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)


class EnergyFunction(ABC):
    """Parent class for various thermodynamic funcitons describing energy needed for CO2 separation"""

    subclasses = {}

    @classmethod
    def register_subclass(cls, function_name: str):
        """creates decorator that automatically registers subclasses
        To use, decorate the child class definition with @DataReader.register_subclass("[name-of-child-class]")
        """

        def decorator(subclass):
            cls.subclasses[function_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(
        cls, function_name: str, params: Optional[Union[None, Dict[str, float]]] = None
    ):
        """Creates a new child class using the function_name (e.g., "constant" or "linear")"""
        if function_name not in cls.subclasses:
            raise ValueError(f"Bad function_name {function_name}")

        return cls.subclasses[function_name](params)

    def __init__(self, params: Union[None, Dict[str, float]]):

        # set private parameter values (values in the function that are not meant to be changed)
        if params is not None:
            if not isinstance(params, dict):
                raise Exception(
                    "params must be a dictionary with string-valued keys and float-valued values"
                )
            for param, value in params.items():
                setattr(self, param, value)

    @abstractmethod
    def evaluate(self):
        pass


@EnergyFunction.register_subclass("constant")
class ConstantEnergyFunction(EnergyFunction):
    """A constant amount of energy needed for separating CO2 from gas stream"""

    def __init__(self, params):
        # initialize parent class __init__
        super().__init__(params)
        self.function_name = "constant"

        logging.info(" Initialized a constant thermodynamic function.")

    def evaluate(self):
        a = self.constant_value
        return a


@EnergyFunction.register_subclass("linear")
class LinearEnergyFunction(EnergyFunction):
    """A constant amount of energy needed for separating CO2 from gas stream"""

    def __init__(self, params):
        # initialize parent class __init__
        super().__init__(params)
        self.function_name = "linear"
        attributes = self.__dict__
        if ("f_min" not in attributes) | ("f_max" not in attributes):
            raise AttributeError(
                "f_min and f_max must be specified as parameters for the linear energy function"
            )
        logging.info(" Initialized a constant thermodynamic function.")

    def evaluate(self, alpha: float):
        if not isinstance(alpha, float):
            raise TypeError(
                f"Method argument alpha must be a float within [0,1]. value passed is {alpha}"
            )
        if (alpha < 0.0) | (alpha > 1.0):
            raise ValueError(f"evaluate argument alpha must be within range [0,1]")

        return self.f_min + alpha * (self.f_max - self.f_min)
