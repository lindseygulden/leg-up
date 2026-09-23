"""Typed configuration for industrial emissions computation."""

# pylint: disable=too-many-instance-attributes, missing-function-docstring

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from projects.ccs_ghg.config import CCSConstants, IndustryInput
from utils.io import yaml_to_dict

if TYPE_CHECKING:
    from projects.ccs_ghg.config import CCSProjectConfig

__all__ = ["IndustryInput", "IndustrialEmissionsConfig"]


@dataclass
class IndustrialEmissionsConfig:
    """Top-level config combining constants with per-industry inputs."""

    constants: CCSConstants
    industries: List[IndustryInput]

    @classmethod
    def from_project_config(
        cls, cfg: "CCSProjectConfig"
    ) -> "IndustrialEmissionsConfig":
        return cls(
            constants=cfg.constants,
            industries=cfg.get_all_industry_inputs(),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "IndustrialEmissionsConfig":
        """read in configuration from dictionary"""
        cc = CCSConstants.from_dict(d["constants"])
        industries = [IndustryInput.from_dict(ind) for ind in d["industries"]]
        return cls(constants=cc, industries=industries)

    @classmethod
    def from_yaml(cls, path: str) -> "IndustrialEmissionsConfig":
        """read in configuration from a yaml file with specified path"""

        return cls.from_dict(yaml_to_dict(path))
