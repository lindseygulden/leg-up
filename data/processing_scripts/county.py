""" reads in county health data, processes it into a single pandas data frame, and writes it out to a csv"""
# Data available from
# https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation


import logging
from pathlib import PosixPath
from typing import Union

import click
import numpy as np
import pandas as pd

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
def county_data(
    excel_input: Union[str, PosixPath], output_filepath: Union[str, PosixPath]
):
    """Reads IHME excel file sheets, extracts 2014 data, merges all cancer types by FIPS, writes to csv"""

    logging.info(" --- Reading county statistics data")

    excel_file = pd.ExcelFile(excel_input)

    # First extract the 'ranked measure' data for each county
    sheet = excel_file.parse("Ranked Measure Data")
    sheet.columns = sheet.loc[0, :]
    sheet = sheet.iloc[1:, :]
    ranked_measure_cols = [
        x
        for x in sheet.columns.values
        if (("% " in x) | ("Rate" in x) | ("Ratio" in x)) and ("95" not in x)
    ]

    risk_factors_df = sheet[["FIPS"] + ranked_measure_cols]

    risk_factors_df.columns = [
        x.lower()
        .replace("%", "pct")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_or_")
        for x in risk_factors_df.columns.values
    ]

    # now extract the 'additional' measure data
    sheet = excel_file.parse("Additional Measure Data")
    sheet.columns = sheet.loc[0, :]
    sheet = sheet.iloc[1:, :]
    cols = [
        x
        for x in sheet.columns.values
        if ("95%" not in x)
        and ("Deaths" not in x)
        and ("#" not in x)
        and (x != "Sample Size")
        and ("Ratio" not in x)
        and (x not in ["State", "County", "Population"])
    ]
    additional_df = sheet[cols]
    additional_df.columns = [
        x.lower()
        .replace(" ", "_")
        .replace("/", "")
        .replace("%", "pct")
        .replace("-", "")
        .replace("<", "lt")
        for x in additional_df.columns.values
    ]
    additional_cols = list(
        additional_df.columns.values[list(additional_df.isna().sum() < 3)]
    )

    additional_df = additional_df[additional_cols]

    # add the additional factors to the to the rest of the risk factors

    risk_df = risk_factors_df.merge(additional_df, on="fips")

    risk_df.dropna(inplace=True)

    # convert ratios to whole numbers and every number to floats
    for c in [x for x in risk_df.columns.values if "ratio" in x]:
        risk_df[c] = [
            int(x.split(":")[0]) if isinstance(x, str) else np.nan for x in risk_df[c]
        ]
    risk_columns = [x for x in risk_df.columns.values if x != "fips"]
    risk_df[risk_columns] = risk_df[risk_columns].astype(float)

    risk_df.to_csv(output_filepath)

    logging.info(
        " --- Consolidated county statistics data written to %s", output_filepath
    )


if __name__ == "__main__":
    county_data()
