# Simple command-line interface tool generation for this repository
Constructed with inspiration from this [RealPython tutorial](https://realpython.com/python-click/#preparing-a-click-app-for-installation-and-use) and this [medium article](https://medium.com/clarityai-engineering/how-to-create-and-distribute-a-minimalist-cli-tool-with-python-poetry-click-and-pipx-c0580af4c026)  showing how to leverage poetry and click to generate simmple command-line-interface capability:

# Requirements
The [click](https://click.palletsprojects.com/en/8.1.x/) library for simple command-line tool generation

The [poetry dependency management](https://python-poetry.org), including adding the following lines in pyproject.toml:
[tool.poetry.scripts]
cli = "cli.run:cli"

# To use 
### In root directory:
```>> cli [sub-command name] [--flags and arguments for subcommand ]```
### For example:
```>> cli wwo --config data/newton_weather.yml --output_path data/results```

# To add new commands/capability:
1. define a sub-command function, [name_of_new_command], in cli_commands.py,
2. link it here:
 cli.add_command(cli_commands.[name_of_new_command])
