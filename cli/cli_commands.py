"""Command-line functions for projects in folder
To add a new functionality to the interface:
    1. Define a new sub command below using click decorators (see, for example, def wwo() below)
"""

from pathlib import Path

import click

from projects.weather.wwo_data_reader import DataReader


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output_path", type=click.Path(file_okay=False, dir_okay=True), required=True
)
def wwo(config: str, output_path: str):
    """Uses command-line arguments to set up a call to the WWO API: assembles data in csv(s)
    Args:
        config: string/path to configuration yaml file
        output_path: path do directory
    Returns:
        None
    """
    # Create a new WWO DataReader object according to details in the config yaml
    d = DataReader.create("wwo", config)

    # Prepare the API query info
    d.prep_query()

    # Obtain data
    d.get_data()

    # Format the data into writeable dataframes
    d.postprocess_data()

    # if output directory does not yet exist, make it
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Write data to the specified output path
    d.write_data(output_path)
