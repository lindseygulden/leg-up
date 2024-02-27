# Simple command-line interface tool generation for this repository
Constructed with inspiration from this [RealPython tutorial](https://realpython.com/python-click/#preparing-a-click-app-for-installation-and-use) and this [medium article](https://medium.com/clarityai-engineering/how-to-create-and-distribute-a-minimalist-cli-tool-with-python-poetry-click-and-pipx-c0580af4c026)  showing how to leverage poetry and click python libraries to generate simple command-line-interface capability.

# Requirements
* The [click](https://click.palletsprojects.com/en/8.1.x/) library for simple command-line tool generation.
* Use of [poetry](https://python-poetry.org) for environment/dependency management.
* Modification of the [pyproject.toml](https://github.com/lindseygulden/leg-up/blob/a27e311bd2c89cca26c7f6d8ebd99aa78f3fad22/pyproject.toml) to include the following lines:
```
[tool.poetry.scripts]
cli = "cli.run:cli"
```

# To use 
### In root directory:
```>> cli [sub-command name] [--flags and arguments for subcommand ]```
### For example:
```>> cli wwo --config data/newton_weather.yml --output_path data/results```

# To add new commands/capability:
1. Define a sub-command function, `name-of-new-command`, in [cli_commands.py](https://github.com/lindseygulden/leg-up/blob/447232f3454ac8a340ad04ae176f81c745009b88/cli/cli_commands.py),
2. Add it to cli in [run.py]():
```
cli.add_command(cli_commands.name-of-new-command])
```
