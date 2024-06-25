"""Command-line script to query the PVWatts v8 API for various lat/lons"""

import logging
from pathlib import PosixPath
from typing import Union

import click
import pandas as pd
import requests

from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
def pvwatts(config: Union[str, PosixPath]):
    """For a series of specified states (or 'sub-state' areas), query PVWatts V8, assemble data"""
    config = yaml_to_dict(config)

    state_df = pd.read_csv(config["state_locations"], index_col=[0])

    info_list = []
    for _, row in state_df.loc[
        state_df.state.isin(config["states_to_get"]), :
    ].iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        state = row["state"]
        logging.info("state %s", state)
        for tilt in config["tilts"]:
            for azimuth in config["azimuths"]:
                info_dict = {}
                # pylint: disable=line-too-long
                query = f"https://developer.nrel.gov/api/pvwatts/v8.json?api_key={config['apikey']}&lat={lat}&lon={lon}&system_capacity={config['kw']}&azimuth={azimuth}&tilt={tilt}&array_type=1&module_type=1&losses=10"
                response = requests.get(query, timeout=20)
                info_dict["ac_annual"] = response.json()["outputs"]["ac_annual"]
                info_dict["lat"] = lat
                info_dict["lon"] = lon
                info_dict["state"] = state
                info_dict["tilt"] = tilt
                info_dict["azimuth"] = azimuth
                info_dict["pv_array_size_kw"] = config["kw"]
                info_list.append(info_dict.copy())

    solar_df = pd.DataFrame(info_list)
    solar_df.to_csv(config["outfile"])
    logging.info("Solar production data written to %s", config["outfile"])


if __name__ == "__main__":
    pvwatts()
