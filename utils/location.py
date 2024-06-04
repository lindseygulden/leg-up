""" Assorted functions for manipulating location/geographic data"""

from typing import Optional

from geopy.geocoders import Nominatim


def city_lat_lon(
    city: Optional[str] = None,
    state: Optional[str] = None,
    zipcode: Optional[str] = None,
    country: Optional[str] = None,
):
    """Returns (lat, lon) coordinates for a string descriptor of a location
    Args:
        city: string specifying city, if available
        state: string specifying state, if available
        zipcode: string specfiying zip code (US) if available
        country: sptring specifying country, if available
    Returns:
        tuple with the latitude and longitude coordinates (in degrees N and E)
    """
    if city is None and state is None and zipcode is None and country is None:
        raise ValueError(
            "At least one of the following keyword arguments must be present: city, state, zipcode, country"
        )
    location_string = " ".join(
        [x for x in [city, state, zipcode, country] if x is not None]
    )
    geolocator = Nominatim(user_agent="myapplication")

    location = geolocator.geocode(location_string)
    return (float(location.raw["lat"]), float(location.raw["lon"]))
