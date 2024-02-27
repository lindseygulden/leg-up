from unittest.mock import mock_open, patch

from utils.io import yaml_to_dict


def test_yaml_to_dict():
    with patch("builtins.open", new_callable=mock_open, read_data="yams: tasty") as m:
        result = yaml_to_dict("mock_file.yaml")

        # did we call the mock open once?
        m.assert_called_once_with("mock_file.yaml", "r", encoding="utf-8")

        # does the result match what we expect?
        assert result == {"yams": "tasty"}
