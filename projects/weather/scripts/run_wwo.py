import click


@click.command()
@click.argument("config_yml", type=click.Path(), required=True)
@click.argument("output_path", type=click.Path(exists=True), required=True)
def get_weather(config_yml: str, output_path: str):
    """Uses command-line arguments to set up a call to the WWO API: assembles data in single csv"""
    config = yaml_to_dict(config_yml)
    print(config)
    # Get list of locations
    locations = config["locations"]

    # Convert dates to list that is usable by the API
    query_start_date = dt.datetime.strptime(config["start_date"].title(), "%d-%b-%Y")
    query_end_date = dt.datetime.strptime(config["end_date"].title(), "%d-%b-%Y")
    dt_list = split_date_range(query_start_date, query_end_date)

    date_chunks = []

    for loc in locations:
        # for every subset of dates
        for i, dt_start in enumerate(dt_list[:-2]):
            dt_end = dt_list[i + 1]
            request_response = query_wwo_api(
                config["entry_point"],
                config["api_key"],
                loc,
                config["frequency"],
                dt_start,
                dt_end,
                config["timeout_seconds"],
            )

            df = pd.DataFrame.from_dict(request_response.json())


if __name__ == "__main__":
    get_weather()
