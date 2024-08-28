""" bespoke utils for processing ccs-related queries"""

import re

import pandas as pd
from cleanco import basename


def clean_client_names(df: pd.DataFrame):
    """bespoke function for cleaning/standardizing client (company) names for ccs query"""
    starting_client_names = sorted(list(df.client_name.unique()))
    client_names = sorted(list(df.client_name.unique()))
    client_names = [
        basename(re.sub(r"[^\w\s]", "", company_name)) for company_name in client_names
    ]

    client_names = [x.split("ON BEHALF OF")[-1] for x in client_names]

    for n in [
        " CORPORATION",
        " LLC",
        " FKA",
        " VENTURES",
        " NORTH AMERICA",
        " LONDON",
        " USA",
        "FORMERLY KNOWN AS",
        "FORMERLY",
        " (FKA",
    ]:
        client_names = [x.split(n)[0] for x in client_names]

    for n in [
        "SERVICES",
        "THE ",
        "COMPANY",
        "INC",
        "GROUP",
        "CRES",
        "NRECA",
        "INFORMAL",
    ]:
        client_names = [x.replace(n, "") for x in client_names]
    client_names = [x.replace("  ", " ") for x in client_names]
    client_names = [x.rstrip().lstrip() for x in client_names]
    # Bespoke replacements
    for key, value in {
        "MUNCIPAL": "MUNICIPAL",
        "DEPT": "DEPARTMENT",
        "ASSOCCIATION": "ASSOCIATION",
        "ISG ON BEHHALF OF": "USA BIOMASS POWER PRODUCERS ALLIANCE",
        "SIGLOBAL": "SITHE GLOBAL",
        "SOUTHERN CALIFORNIA EDISON": "EDISON INTERNATIONAL",
    }.items():
        client_names = [x.replace(key, value) for x in client_names]

    # BESPOKE REPLACEMENTS
    for company in [
        "EXXON",
        "MARATHON",
        "SHELL",
        "CHEVRON",
        "DENBURY",
        "DOMINION",
        "DOW",
        "AMERICAN ELECTRIC POWER",
        "BABCOCK WILCOX",
        "CONSTELLATION ENERGY",
        "LINDE",
        "SASOL",
        "HOLCIM",
        "NEXTERA",
        "PORTLAND CEMENT ASSOCIATION",
        "SEMPRA",
        "SIEMENS",
        "TALLGRASS",
        "TECO",
        "GREAT PLAINS INSTITUTE",
        "TOYOTA",
        "VISTRA",
        "ALSTOM",
        "ZURICH",
        "ALLETE",
        "BIOMASS POWER ASSOCIATION",
        "PRAIRIE STATE GENERATING",
        "PEABODY",
        "CHAMBER OF COMMERCE",
    ]:
        client_names = [company if company in x else x for x in client_names]

    client_names = [
        x.replace("GE ENERGY", "GE/ALSTOM").replace("ALSTOM", "GE/ALSTOM")
        for x in client_names
    ]
    client_name_rename_dict = dict(zip(starting_client_names, client_names))
    df["client_rename"] = [client_name_rename_dict[x] for x in df.client_name]

    return df
