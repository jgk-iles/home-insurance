"""Tests for name matching function
"""

import sys
import pytest
import pandas as pd
from src.preprocessing import match_participant_id, _combine_name_cols

sys.path.append("..")

# Names from campaign table
LEFT_NAMES = pd.Series(
    [
        "Mr. Dale Coles",
        "Joel Allen",
        "Mr. Craig Marc Davis",
        "Mr. Brandon Thornton",
        "Miss Brett Carol Fletcher",
        "Dr. Christopher Barker",
        "Ms. Jeremy Ward",
        "Mr. Karl Gavin Cox",
        "Hugh Stacey Armstrong",
        "Mr. Mathew Andrew Fox",
        "Aaron Arthur Griffiths",
        "Carl Brandon Dale",
        "Miss Christian Barton",
        "Dr. Declan Hart",
        "Barry Brown",
        "Mathew Kevin Brown",
        "Mr. Raymond Dylan Nicholson",
        "Mr. Benjamin Bartlett",
        "Bradley Rose",
        "Ross Carr",
    ]
)

# Combined / cleaned names from mortgage table
RIGHT_NAMES = pd.Series(
    [
        "Mr. Dale Coles",
        "Joel Allen",
        "Mr. Craig Davis",
        "Mr. Brandon Thornton",
        "Miss Brett Fletcher",
        "Dr. Christopher Barker",
        "Ms. Jeremy Ward",
        "Mr. Karl Cox",
        "Mrs. Hugh Armstrong",
        "Mr. Mathew Fox",
        "Aaron Griffiths",
        "Carl Dale",
        "Christian Barton",
        "Declan Hart",
        "Mr. Barry Brown",
        "Mathew Brown",
        "Mr. Raymond Nicholson",
        "Mr. Benjamin Bartlett",
        "Bradley Rose",
        "Ms. Ross Carr",
    ]
)


@pytest.fixture
def campaign_df() -> pd.DataFrame:
    df = pd.read_csv("data/Campaign.csv")
    df = df.iloc[:30][["participant_id", "name_title", "first_name", "last_name"]]
    df["full_name"] = _combine_name_cols(df.name_title, df.first_name, df.last_name)
    return df[["participant_id", "full_name"]]


@pytest.fixture
def mortgage_df() -> pd.DataFrame:
    df = pd.read_csv("data/Mortgage.csv")
    df = df.iloc[:30][["full_name"]]
    return df


@pytest.fixture
def mortgage_df_with_id() -> pd.DataFrame:
    mortgage_names = pd.read_csv("data/Mortgage.csv")["full_name"].iloc[:30]
    participant_ids = pd.read_csv("data/Campaign.csv")["participant_id"].iloc[:30]
    concat_df = pd.concat([participant_ids, mortgage_names], axis=1)
    return concat_df


def test_match_names(campaign_df, mortgage_df, mortgage_df_with_id):
    """Test for matching names between tables."""
    matched_df = match_participant_id(campaign_df, mortgage_df)
    pd.testing.assert_frame_equal(matched_df, mortgage_df_with_id)
