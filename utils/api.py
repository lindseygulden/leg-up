"""utility functions for working with apis"""
import requests


def api_authenticate(
    authentication_endpoint,
    username,
    apikey,
    username_key="username",
    password_key="password",
):
    """logs in to the api using provided authentication endpoint and credentials"""
    authenticated_session = requests.Session()

    authenticated_session.post(
        authentication_endpoint,
        json={
            username_key: username,
            password_key: apikey,
        },
        timeout=60,
    )

    return authenticated_session
