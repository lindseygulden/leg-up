import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from tools.data_reader.data_reader import DataReader

# TODO figure out how to use pytest.fixtures for MockDataReader class


class TestDataReader(unittest.TestCase):
    def test_register_subclass(self):
        @DataReader.register_subclass("mock")
        class MockDataReader(DataReader):
            def __init__(self, params):
                pass

            def get_data(self):
                pass

            def postprocess_data(self):
                pass

            def prep_query(self):
                pass

        self.assertIn("mock", DataReader.subclasses)
        self.assertEqual(DataReader.subclasses["mock"], MockDataReader)

    def test_instantiate_mock_subclass(self):
        @DataReader.register_subclass("mock")
        class MockDataReader(DataReader):
            def __init__(self, params):
                self.params = params
                pass

            def get_data(self):
                pass

            def postprocess_data(self):
                pass

            def prep_query(self):
                pass

        fake_params = {}

        self.assertIsInstance(
            DataReader.create(data_source="mock", params=fake_params), MockDataReader
        )

    @patch("pandas.DataFrame.to_csv")
    def test_write_data(self, mock_to_csv):
        @DataReader.register_subclass("mock")
        class MockDataReader(DataReader):
            def __init__(self, params):
                pass

            def get_data(self):
                pass

            def postprocess_data(self):
                pass

            def prep_query(self):
                pass

        fake_params = {}
        test_reader = MockDataReader(fake_params)
        data_key = "yellow_polka_dot"
        test_reader.data_dict = {
            "yellow_polka_dot": pd.DataFrame(
                {"itty": "teeny", "bitty": "weeny"}, index=[0]
            )
        }
        test_reader.output_file_suffix = "bikini"
        output_directory = "test_output_directory"
        test_reader.write_data(output_directory)

        mock_to_csv.assert_any_call(
            Path(output_directory)
            / Path(f"{str(data_key)}_{test_reader.output_file_suffix}.csv")
        )

    def test_rename_columns(self):
        @DataReader.register_subclass("mock")
        class MockDataReader(DataReader):
            def __init__(self, params):
                pass

            def get_data(self):
                pass

            def postprocess_data(self):
                pass

            def prep_query(self):
                pass

        fake_params = {}
        test_reader = MockDataReader(fake_params)
        test_reader.data_renaming_dict = {
            "column_names": {"itty": "very_itty", "bitty": "very_bitty"}
        }

        test_df = pd.DataFrame({"itty": "teeny", "bitty": "weeny"}, index=[0])
        correct_output_df = pd.DataFrame(
            {"very_itty": "teeny", "very_bitty": "weeny"}, index=[0]
        )
        revised_df = test_reader._rename_columns(test_df)
        assert_frame_equal(revised_df, correct_output_df)


if __name__ == "__main__":
    unittest.main()
