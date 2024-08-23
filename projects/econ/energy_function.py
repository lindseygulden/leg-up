""" Abstract base class for thermodynamic function defining energy needed to separate CO2 """

import logging
from abc import ABC, abstractmethod
from pathlib import PosixPath
from typing import List, Union, Dict

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
        cls,
        function_name: str,
        constant: float,
        other_params: Union[None, Dict[str, float]],
        variables: Union[None, Dict[str, float]],
    ):
        """Creates a new child class using the function_name (e.g., "constant" or "linear")"""
        if function_name not in cls.subclasses:
            raise ValueError(f"Bad function_name {function_name}")

        return cls.subclasses[function_name](constant, other_params, variables)

    def __init__(
        self,
        constant: float,
        other_params: Union[None, Dict[str, float]],
        variables: Union[None, Dict[str, float]],
    ):
        self.__constant = constant
        # set private parameter values (values in the function that are not meant to be changed)
        if other_params is not None:
            for param, value in other_params.items():
                setattr(self, "__" + param, value)
        # initialize variables
        if variables is not None:
            for var, initial_value in variables.items():
                setattr(self, var, initial_value)

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError("Abstract method in the base class")


@EnergyFunction.register_subclass("constant")
class ConstantEnergyFunction(EnergyFunction):
    """A constant amount of energy needed for separating CO2 from gas stream"""

    def __init__(self, constant: float, params=None, variables=None):
        # initialize parent class __init__
        super().__init__(constant, params, variables)
        self.function_name = "constant"

        logging.info(" Initialized a WWO weather-data reader.")

    def evaluate(self):
        return self.__constant
