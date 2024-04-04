""" Pulls in age-adjusted cancer incidence rates from csv-formatted NIH/CDC county-by county data"""

# County scale data for various cancers downloaded to a local directory
# data in format available from https://www.statecancerprofiles.cancer.gov/incidencerates/

import logging
import os
from pathlib import Path, PosixPath
from typing import Union

import click
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--input_data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    "--output_filename", type=click.Path(file_okay=True, dir_okay=False), required=True
)
def cancer_incidence(input_data_dir: Union[str, PosixPath], output_filename: str):
    """Reads raw csvs, lightly processes them, merges all types of cancer by FIPS, writes df to csv"""

    logging.info(" --- Reading raw incidence data ---")
    files = os.listdir(Path(input_data_dir))
    files = [
        x
        for x in files
        if (".csv" in x) and ("._" not in x) and (output_filename not in x)
    ]
    type_dict = dict(zip([x.split("_incidence", 1)[0] for x in files], files))

    cancer_df = pd.DataFrame()

    # Iterate through each age-adjusted incidence file in the input_data_dir
    for cancer_type, filename in type_dict.items():
        df = pd.read_csv(f"{input_data_dir}/{filename}")
        df.columns = df.loc[7, :]
        logging.info("--- Reading and processing data for %s ---", cancer_type)
        keep_columns = [
            " FIPS",
            "Age-Adjusted Incidence Rate([rate note]) - cases per 100,000",
            "Recent 5-Year Trend ([trend note]) in Incidence Rates",
        ]
        # get rid of the useless rows at the top of the CSV
        df = df.loc[9:, keep_columns]
        df.dropna(inplace=True)
        # make columns more informative/easier to type
        df.columns = [
            "fips",
            f"age_adj_{cancer_type}_incidence_rate_cases_per_100000",
            f"recent_5yr_trend_in_{cancer_type}_incidence_rates",
        ]

        # for rows with insufficient data for that county, replace string value with np.nan
        df.replace(
            {
                "*": np.nan,
                "* ": np.nan,
                "data not available": np.nan,
                "data not available ": np.nan,
            },
            inplace=True,
        )
        # Turn strings into floats
        for c in [
            f"age_adj_{cancer_type}_incidence_rate_cases_per_100000",
            f"recent_5yr_trend_in_{cancer_type}_incidence_rates",
        ]:
            df[c] = df[c].astype(float)

        # append this cancer-type data to the dataframe
        if len(cancer_df) == 0:
            cancer_df = df.copy(deep=True)
        else:
            cancer_df = cancer_df.merge(df, how="outer", on="fips")

    cancer_df.to_csv(f"{input_data_dir}/{output_filename}")
    logging.info(
        "--- Processed data written to %s/%s ---", input_data_dir, output_filename
    )
    return cancer_df


if __name__ == "__main__":
    cancer_incidence()
