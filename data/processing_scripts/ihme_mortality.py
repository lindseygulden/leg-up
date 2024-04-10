""" Pulls in age-adjusted mortality rates from IHME-formatted Excel file"""

# County-scale data for mortality from various causes. Can be used for:
# age-adjusted cancer mortality data for 29 cancers
# age-adjusted cardiovascular disease
# must be in the format of data available from https://ghdx.healthdata.org

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
def ihme_mortality_rates(
    excel_input: Union[str, PosixPath], output_filepath: Union[str, PosixPath]
):
    """Reads IHME excel file sheets, extracts 2014 data, merges disease types by FIPS, writes to csv"""

    logging.info(" --- Reading IHME Excel file containing mortality-rate data")

    excel_file = pd.ExcelFile(excel_input)

    # link excel file's sheet names to cancer-type name
    disease_type_sheet_dict = dict(
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
    all_diseases_df = pd.DataFrame()

    # loop through mortality data for all disease types, extract data
    for disease_type, sheet_name in disease_type_sheet_dict.items():
        sheet = excel_file.parse(sheet_name)
        sheet.columns = sheet.iloc[0, :]
        # extract only the 2014 column (note that the formatted file contains data for several years since 1980)
        disease_df = sheet[["FIPS", "Mortality Rate, 2014*"]]
        disease_df.columns = ["fips", f"{disease_type}_2014"]

        disease_df = disease_df.iloc[3:, :]

        disease_df[f"{disease_type}_2014"] = [
            float(x.split(" (")[0]) if isinstance(x, str) else x
            for x in disease_df[f"{disease_type}_2014"]
        ]

        # fix the zero-truncation of the FIPS numbers for counties
        disease_df.fips = [
            zero_pad(x, front_or_back="front", max_string_length=5)
            for x in disease_df.fips
        ]

        disease_df.dropna(inplace=True)

        # get rid of the rows for state-wide data (b/c we only want county data)
        disease_df = disease_df.loc[[x[:3] != "000" for x in disease_df["fips"]]]

        # merge mortality data for this disease to main dataframe
        if len(all_diseases_df) == 0:
            all_diseases_df = disease_df.copy(deep=True)
        else:
            all_diseases_df = all_diseases_df.merge(disease_df, on="fips", how="outer")

    all_diseases_df.to_csv(output_filepath)
    logging.info("--- Processed data written to %s", output_filepath)
    return all_diseases_df


if __name__ == "__main__":
    ihme_mortality_rates()
