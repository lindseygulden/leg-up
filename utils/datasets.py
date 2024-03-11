"""this file contains functions that obtain and/or lightly process commonly used datasets"""
import pandas as pd

from utils.data import zero_pad

US_CENSUS_COUNTY_DATA_WEB_ADDRESS = (
    "http://www2.census.gov/geo/docs/reference/codes/files/national_county.txt"
)


def get_us_state_to_abbr_dict():
    """returns conversion dictionary. Credit/thanks to Roger Allen: https://gist.github.com/rogerallen/1583593"""
    return {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
        "District of Columbia": "DC",
        "American Samoa": "AS",
        "Guam": "GU",
        "Northern Mariana Islands": "MP",
        "Puerto Rico": "PR",
        "United States Minor Outlying Islands": "UM",
        "U.S. Virgin Islands": "VI",
    }


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

    us_state_to_abbr = get_us_state_to_abbr_dict()
    abbr_to_us_state = {a: s for s, a in us_state_to_abbr.items()}
    # expand the state abbreviation for easier merging
    county_df["state"] = [abbr_to_us_state[x] for x in county_df.state_abbr]

    # housecleaning
    county_df.drop("h", inplace=True, axis=1)

    return county_df
