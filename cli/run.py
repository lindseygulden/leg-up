""" Core access point for command-line-interface tool generation for this repository"""

# Constructed with inspiration from RealPython tutorial and a medium article showing how
# to leverage poetry and click to generate simmple command-line-interface capability:
# https://realpython.com/python-click/#preparing-a-click-app-for-installation-and-use
# https://medium.com/clarityai-engineering/how-to-create-and-distribute-a-minimalist-cli-tool-with-python-poetry-click-and-pipx-c0580af4c026
#
# Note that this implementation depends on use of poetry, including the following lines in pyproject.toml:
# [tool.poetry.scripts]
# cli = "cli.run:cli"
#
# To use, in root directory:
# >> cli [sub-command name] [--flags and arguments for subcommand ]
# For example:
# >> cli wwo --config data/newton_weather.yml --output_path data/results
#
# To add new commands/capability:
# 1. define a sub-command function, [name_of_new_command], in cli_commands.py,
# 2. link it here:
# cli.add_command(cli_commands.[name_of_new_command])

import click

from cli import cli_commands


@click.group()
def cli():
    """Core access point for cli tools in this repository"""


# Add new sub commands here
cli.add_command(cli_commands.wwo)
