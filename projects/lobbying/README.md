
# README
Files written in Fall 2024 by Lindsey Gulden [lindsey@legupdata.com](mailto:lindsey@legupdata.com). 
This work was done in collaboration with the [Science Roundtable on Carbon Capture and Storage](https://www.capturethetruth.org).

This directory contains python, YAML, and Jupyter files that together are used to query, compile, postprocess, and analyze [Lobbying Disclosure Act (LDA) Reports](https://lda.senate.gov/system/public/).

[Linked here is a written summary](https://docs.google.com/document/d/1BKStXzkElu1F4sogFl9wSw8_8woJIqiMn3FdKy-e0to/edit?usp=sharing) of key conclusions from efforts to quantify lobbying of the federal government regarding carbon capture and storage (CCS). A tabular summary of CCS-related lobbying of the US federal government can be found [here](https://docs.google.com/spreadsheets/d/1pbzg_OYfu8rRQgCW2nRYDXkRrcCdDakaYELiH4j7B4U/edit?usp=sharing).

Note that when looking for lobbying disclosures tied to CCS we looked both for descriptions of CCS as well as closely related topics (e.g., enhanced oil recovery (EOR), 'clean hydrogen' or 'blue hydrogen' (i.e., hydrogen made with methane), 'class VI well permitting reform', carbon dioxide pipelines, etc.)

## In addition to these files, what else might I need?
* We use pyenv for our virtual environment [poetry](https://python-poetry.org) as our dependency manager. The [poetry.lock](https://github.com/lindseygulden/leg-up/blob/main/poetry.lock) and [pyproject.toml](https://github.com/lindseygulden/leg-up/blob/main/pyproject.toml) files contain requirements/module versions and settings (although, because this is not the only code in this repo, modules needed for this CCS lobbying code are a subset of the modules within the poetry.lock file.)
* To use these files, as-is or in modified form, you'll need to register ([form linked here](https://lda.senate.gov/api/register/)) for an LDA API key and login ID. Add your login id and API key to the relevant configuration YAML file. (See examples at [projects/lobbying/config](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/config).)
* Several of the functions here leverage [utility functions](https://github.com/lindseygulden/leg-up/blob/main/utils) in a separate module of this repo.
* If you'd like to modify and/or run the postprocessing logic yourself, here you will find a compiled CCS lobbying dataset (i.e., the result of running compile_results.py on query results from lda_query.py using config_ccs_lda.yml)
* If you'd like to work with the postprocessed dataset generated with our code (which also underlies the linked tables and text summary above) here you will find a postprocessed CCS lobbying dataset (i.e., the result of running postprocess.py on compiled results, both using config_ccs_lda.yml)

## What's in this directory?
* [General configuration YAML files](https://github.com/lindseygulden/leg-up/tree/main/projects/lobbying#configuration-information-stored-in-yaml-files) to enable consistent and traceable querying, compilation, and postprocessing
* [Search-term YAML files](https://github.com/lindseygulden/leg-up/tree/main/projects/lobbying#search-term-lists-for-lda-queries-are-also-stored-in-yaml-files), which contain a record of terms used for LDA API queries.
* [Definition YAML files](https://github.com/lindseygulden/leg-up/tree/main/projects/lobbying#additional-definitions-general-information-settings-name-replacements-and-sector-assignments) in which we specify key indicators of topics (CCS, in this case), organization names, and sector assignments.
* [Utility functions used in command-line scripts](https://github.com/lindseygulden/leg-up/tree/main/projects/lobbying#utility-functions-that-support-the-command-line-scripts-for-querying-compiling-and-postprocessing-lda-reports).
* [Three command-line scripts, to be run in sequence](https://github.com/lindseygulden/leg-up/tree/main/projects/lobbying#three-scripts-for-running-the-whole-enchilada). The scripts (1) query the LDA API; (2) compile results of the LDA queries; and (3) postprocess the compiled results to identify lobbying activities of interest.
, and generating analysis figure/stables.
* [A Jupyter notebook](https://github.com/lindseygulden/leg-up/tree/main/projects/lobbying#figures-and-tables-for-presenting-results) for generating figures and doing basic data processing, reporting

### Configuration information stored in YAML files
Configuration files, found in (this directory)[https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/config/], house paths to data files, API keys, postprocessing files and parameters, etc.

[The primary CCS querying configuration file can be found here](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/config/config_ccs_lda.yml) This file contains configuration settings for querying, compiling, and postprocessing lobbying disclosures that may deal with CCS. Like other config files, it sets up (and tracks) the configurations necessary when running command-line scripts lda_query.py, compile_results.py, and postprocess.py, in this case for the topic of carbon capture and storage.

[A configuration file for querying, compiling, and processing all lobbying done by 'CCS Big Spenders' can be found here](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/config/config_biggest_ccs_spenders_organizations.yml) 

[A configuration file for querying, compiling, and processing lobbying done by organizations associated with the Iron and Steel industries can be found here](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/config/config_steel_iron_organizations.yml) 

### Search-term lists for LDA queries are also stored in YAML files
Search term lists, found in [this directory](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/search_terms/), are YAML-formatted lists of query terms for use by [lda_query.py](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/lda_query.py). 
Note that search term lists are either for querying the 'client_name' or 'filing_specific_lobbying_issues' fields of the LDA reports API. When a query to be submitted to the LDA API will be looking at terms within 'client_name' (that is, the lobbying client that is ultimately responsible for the lobbying of the US federal government), the lists are lists of organization names or terms that would appear within the client name. When the query is to be used with the API field 'filing_specific_lobbying_issues', which supports complex text searching, the search-term yaml files contain lists of specially formatted strings (to be joined by [lda_query.py](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/lda_query.py) with an 'OR').

#### CCS lobbying search term lists:
Note that, to get all lobbying disclosure reports for these three lists of search terms, we ran the lda_query.py script three times, changing the config_ccs_lda.yml path each time.
* [Terms that are typicially part of a description of CCS or a closely related topic ('clean hydrogen', 'hydrogen hubs', 'class VI well permitting reform', EOR, etc.)](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/search_terms/search_term_list_ccs_description.yml)
* [Names of US Congress Bills/Laws that deal with CCS](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/search_terms/search_term_ccs_bills_by_name.yml)
* [Search strings for bill numbers of bills and laws that dealt mostly or exclusively with CCS](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/search_terms/search_term_ccs_billnos.yml)

#### Other search term lists for lobbying disclosure reports used in this analysis:
* [Organization names, nicknames, etc. for the 'CCS Big Spenders'](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/search_terms/search_terms_ccs_big_spender_orgs.yml). Used with [this config file](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/config/config_biggest_ccs_spenders_organizations.yml) to obtain all lobbying--not just CCS lobbying--done by the organizations reponsible for the most CCS-related lobbying.
* [Organization names, nicknames, etc. for iron-and-steel industry organizations](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/search_terms/search_terms_steel_iron_orgs.yml). Used with [this config file](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/config/config_steel_iron_organizations.yml) to obtain all lobbying--not just CCS lobbying--done by organizations in the iron and steel sector.

### Additional definitions, general information, settings, name replacements, and sector assignments:
The YAML files in the [definitions](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/definitions/) folder contain a range of other information used for compilation, postprocessing, and analysis.
#### Information used for first pass of handling raw lobbying-activity data:
* [Consolidation of the various abbreviations for Public Laws as well as House and Senate bills seen in the lobbying activity descriptions can be found in bill_abbreviations.yml](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/definitions/bill_abbreviations.yml) 
* [Information for handling organization names in compile_results.py can be found in organization_name_handling.yml](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/definitions/organization_name_handling.yml)
* [A dual-purpose--sorry--file that contains a list of terms that, if present in an organization's name or description, result in the organization's lobbying activities being discarded](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/definitions/sector_descriptions.yml). The second purpose of this was for me to make a first pass at generating sectors_updated.yml (which I ended up hand tuning using research about various organization's sectors, purposes, mergers, etc.)

#### Information used in post-processing:
* [The primary postprocessing-configuration file (ccs_postproc_specifications.yml)](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/definitions/ccs_postproc_specifications.yml). This YAML file contains information that is used to attribute meaning to lobbying activities and identifying activities of interest.
* [party_control_congress.yml, which maps the Congress number (e.g., 118th) to the political party in control of the House, Senate, and White House](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/definitions/party_control_congress.yml)
* [A mapping of organiztion names to detailed sectors can be found in sectors_updated.yml](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/definitions/sectors_updated.yml). The first pass was done by machine; the second (and third) pass and reassignments were done manually.
* [A apping of detailed sectors to 'lightly lumped' sectors and 'very lumped' sectors can be found in lumped_sectors.yml](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/definitions/lumped_sectors.yml)
* [A replacement dictionary for organization names that appear in the compilation generated by compile_results.py can be found in company_name_replacements.yml](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/definitions/company_name_replacements.yml). Note that this is the main way that nicknames, mergers, renames, subsidiaries, etc. are handled.

### Utility functions that support the command-line scripts for querying, compiling, and postprocessing LDA reports
[Bespoke utility functions for lda_query.py](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/lda_query_utils.py)

[Utility functions used, to some extent, in both compile_results.py and postprocess.py](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/postproc_utils.py)

### Three scripts for running the whole enchilada
1. [Command-line script to query the LDA API](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/lda_query.py). To use (from root directory command line):
```
>> python3 projects/lobbying/lda_query.py --config [path to config.yml] --output_dir [path to directory for results]
```
2. [Command-line script to compile all results from querying LDA API](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/lda_query.py). To use (from root directory command line):
```
>> python3 projects/lobbying/compile_results.py --config [path to config.yml] --input_dir [path to directory containing filing csvs ONLY] --output_file [path to filename where compiled results will be written, in csv form]
```
3. [Command-line script to postprocess the compiled LDA-API query results](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/lda_query.py). To use (from root directory command line):
```
>> python3 projects/lobbying/postprocess.py --config [path to config.yml] --oinput_file [path to filename where compiled results are stored] --output_file [path to filename where postprocessed results will be written, in csv form]
```

### Figures and tables for presenting results
[Jupyter notebook for constructing figures and post-processing](https://github.com/lindseygulden/leg-up/blob/main/projects/lobbying/analysis.ipynb). You'll need to modify the variables linking to postprocessed results and the postprocessing configuration file within the jupyter notebook.