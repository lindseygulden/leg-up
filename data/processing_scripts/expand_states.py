"""script to expand state data to include more than one point for CA and TX"""

import logging

import click
import pandas as pd

logging.basicConfig(level=logging.INFO)

# lat/lon data from https://developers.google.com/public-data/docs/canonical/states_csv
# added lat/lon of Los Angeles and San Francisco, CA and Midland and Houston, TX (according to wikipedia);
# gave them abbreviated names and long names for merging with rate, carbon intensity, etc. data


@click.command()
def states():
    """Script to modify lat/lon data to include expansions for CA/TX. Created for documentation sake"""
    state_df = pd.read_csv("/Volumes/Samsung_T5/data/us_geo/state_lat_lon.csv")
    state_df = pd.concat(
        [
            state_df,
            pd.DataFrame(
                {
                    "state": ["CALA", "CASF", "TXW", "TXE"],
                    "latitude": [34.0549, 37.7749, 31.9973, 29.7604],
                    "longitude": [-118.2426, -122.4194, -102.0779, -95.3698],
                    "name": [
                        "California Los Angeles",
                        "California San Francisco",
                        "Texas West",  # midland
                        "Texas East",  # houston
                    ],
                }
            ),
        ]
    )

    state_df.to_csv(
        "/Volumes/Samsung_T5/data/us_geo/state_lat_lon_with_CA_TX_expansions.csv"
    )


if __name__ == "__main__":
    states()
