"""this file contains functions that obtain and/or lightly process commonly used datasets"""
import pandas as pd

from data.general.us_state_abbrev import abbrev_to_us_state
from utils.data import zero_pad

US_CENSUS_COUNTY_DATA_WEB_ADDRESS = (
    "http://www2.census.gov/geo/docs/reference/codes/files/national_county.txt"
)


def get_county_df(web_location_of_file: str = US_CENSUS_COUNTY_DATA_WEB_ADDRESS):
    """download and assemble a pandas dataframe containing FIPS codes and names for all US counties"""
    # download text data
    county_df = pd.read_csv(web_location_of_file, header=None)
    county_df.columns = ["state_abbr", "state_fp", "county_fp", "county_name", "h"]

    # convert two-digit state code and three-digit county code into zero-padded strings
    county_df["state_fp"] = [
        zero_pad(x, max_string_length=2) for x in county_df.state_fp
    ]
    county_df["county_fp"] = [
        zero_pad(x, max_string_length=3) for x in county_df.county_fp
    ]

    # assemble fips
    county_df["fips"] = [s + c for s, c in zip(county_df.state_fp, county_df.county_fp)]

    # delete the word 'County' from all of the county names
    county_df["county_name"] = [x.replace(" County", "") for x in county_df.county_name]

    # expand the state abbreviation for easier merging
    county_df["state"] = [abbrev_to_us_state[x] for x in county_df.state_abbr]

    # housecleaning
    county_df.drop("h", inplace=True, axis=1)

    return county_df
