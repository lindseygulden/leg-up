""" Pulls in age-adjusted cancer mortality rates from IHME Excel file"""

# County-scale data for age-adjusted cancer mortality data for 29 cancers
# data available from https://ghdx.healthdata.org

import logging
from pathlib import PosixPath
from typing import Union

import click
import pandas as pd

from utils.data import zero_pad

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--excel_input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output_filepath", type=click.Path(file_okay=True, dir_okay=False), required=True
)
def cancer_mortality(
    excel_input: Union[str, PosixPath], output_filepath: Union[str, PosixPath]
):
    """Reads IHME excel file sheets, extracts 2014 data, merges all cancer types by FIPS, writes to csv"""

    logging.info(" --- Reading IHME Excel file containing cancer-mortality-rate data")

    excel_file = pd.ExcelFile(excel_input)

    # link excel file's sheet names to cancer-type name
    cancer_type_sheet_dict = dict(
        zip(
            [
                x.replace(" cancer", "")
                .lower()
                .replace("&", "and")
                .replace("-", "")
                .replace(" ", "_")
                for x in excel_file.sheet_names
            ],
            excel_file.sheet_names,
        )
    )
    # initialize the output dataframe
    all_cancers_df = pd.DataFrame()

    # loop through cancer mortality data for all cancer types, extract data
    for cancer_type, sheet_name in cancer_type_sheet_dict.items():
        sheet = excel_file.parse(sheet_name)
        sheet.columns = sheet.iloc[0, :]
        # extract only the 2014 column (note that the formatted file contains data for several years since 1980)
        cancer_df = sheet[["FIPS", "Mortality Rate, 2014*"]]
        cancer_df.columns = ["fips", f"{cancer_type}_2014"]

        cancer_df = cancer_df.iloc[3:, :]

        cancer_df[f"{cancer_type}_2014"] = [
            float(x.split(" (")[0]) if isinstance(x, str) else x
            for x in cancer_df[f"{cancer_type}_2014"]
        ]

        # fix the zero-truncation of the FIPS numbers for counties
        cancer_df.fips = [
            zero_pad(x, front_or_back="front", max_string_length=5)
            for x in cancer_df.fips
        ]

        cancer_df.dropna(inplace=True)

        # get rid of the rows for state-wide data (b/c we only want county data)
        cancer_df = cancer_df.loc[[x[:3] != "000" for x in cancer_df["fips"]]]

        # merge mortality data for this cancer type to main dataframe
        if len(all_cancers_df) == 0:
            all_cancers_df = cancer_df.copy(deep=True)
        else:
            all_cancers_df = all_cancers_df.merge(cancer_df, on="fips", how="outer")

    all_cancers_df.to_csv(output_filepath)
    logging.info("--- Processed data written to %s", output_filepath)
    return all_cancers_df


if __name__ == "__main__":
    cancer_mortality()
