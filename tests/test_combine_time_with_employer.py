import pytest
from pathlib import Path

import pandas as pd

from src.preprocessing import combine_time_with_employer


@pytest.fixture
def time_with_employer_columns():
    """Returns columns relating to time with employer from test Mortgage.csv.

    Returns:
        pandas.DataFrame: DataFrame containing just years and months with employer columns.
    """
    path = Path(__file__).resolve().parent / "testdata" / "Mortgage.csv"
    mort_df = pd.read_csv(path)
    return mort_df[["years_with_employer", "months_with_employer"]]


@pytest.fixture
def combined_months_with_employer():
    correct_months = [
        246,
        337,
        173,
        390,
        42,
        47,
        30,
        29,
        52,
        1,
        55,
        68,
        35,
        4,
        46,
        68,
        31,
        20,
        48,
        162,
        86,
        4,
        109,
        93,
        88,
        11,
        10,
        205,
        121,
        24,
    ]
    return pd.Series(correct_months)


def test_combine_time_with_employer(
    time_with_employer_columns, combined_months_with_employer
):
    time_in_months = time_with_employer_columns.apply(
        lambda x: combine_time_with_employer(
            x["years_with_employer"], x["months_with_employer"]
        ),
        axis=1,
    )
    print(type(time_in_months))
    print(type(combined_months_with_employer))
    pd.testing.assert_series_equal(time_in_months, combined_months_with_employer)
