"""Preproccessing functions"""

from typing import Iterable
import difflib
import re
import requests
import pandas as pd
import numpy as np

# Exchange rate API key
API_KEY = "336451b9507d5a9829f635f5"

FREQUENCIES = {
    "month": 12.0,
    "pw": 52.0,
}


def _combine_name_cols(
    name_title: pd.Series, first_name: pd.Series, last_name: pd.Series
) -> pd.Series:
    """Combines name columns from campaign table into single full name.

    Args:
        name_title (pd.Series): Title column.
        first_name (pd.Series): First name column.
        last_name (pd.Series): Last name column.

    Returns:
        pd.Series: Combined full name.
    """
    name_title = name_title.fillna("").str.strip()
    first_name = first_name.fillna("").str.strip()
    last_name = last_name.fillna("").str.strip()
    full_name = name_title + " " + first_name + " " + last_name
    return full_name.str.strip()


def match_participant_id(
    campaign_df: pd.DataFrame,
    mortgage_df: pd.DataFrame,
    campaign_name: str = "full_name",
    mortgage_name: str = "full_name",
    participant_id: str = "participant_id",
) -> pd.DataFrame:
    """Returns a copy of the Mortgage DataFrame with a participant_id column
    based on the Campaign table

    Args:
        campaign_df (pd.DataFrame):
        mortgage_df (pd.DataFrame): _description_
        campaign_name (str, optional): _description_. Defaults to "full_name".
        mortgage_name (str, optional): _description_. Defaults to "full_name".
        participant_id (str, optional): _description_. Defaults to "participant_id".

    Returns:
        pd.DataFrame: _description_
    """
    mort_df = mortgage_df.copy()
    mort_cols = mort_df.columns.to_list()
    mortgage_ids = []
    for name in mortgage_df[mortgage_name]:
        matches = difflib.get_close_matches(name, campaign_df[campaign_name], n=1)
        if matches:
            idx = campaign_df.loc[
                campaign_df[campaign_name].isin(matches), participant_id
            ].iloc[0]
            mortgage_ids.append(idx)
        else:
            mortgage_ids.append(np.nan)
    mort_df["participant_id"] = mortgage_ids
    return mort_df[["participant_id"] + mort_cols]


def _get_conversion_rates(base_currency: str) -> float:
    try:
        response = requests.get(
            f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/{base_currency}"
        )
    except requests.exceptions.RequestException as error:
        raise SystemExit from error
    else:
        conversion_rates = response.json()["conversion_rates"]
        return conversion_rates


def parse_salary(salary_string: str, exchange_rates: dict) -> float:
    """Parses salary string to return yearly salary in Pounds.

    Args:
        salary_string (str): String description of customer salary.
        exchange_rates (dict, optional): Dictionary of exchange rates.

    Returns:
        float: Yearly salary in British Pounds.
    """
    # Get amount
    try:
        amounts = re.findall(r"\d+\.?\d+", salary_string)
        if len(amounts) < 1:
            return np.nan
        amounts = [float(a) for a in amounts]
        amount = np.mean(amounts)
    except ValueError:
        # Return np.nan if no number found in string
        return np.nan

    # Get currency
    try:
        currency = re.search(r"£|[A-Z]+", salary_string).group()
        if currency == "£":
            currency = "GBP"
    except AttributeError:
        # Set to pounds if no currency found
        currency = "GBP"

    # Get frequency
    try:
        freq = re.findall(r"[a-z]+", salary_string)[-1]
        if freq in list(FREQUENCIES.keys()):  # TODO: Sort out why this is linted
            freq = FREQUENCIES[freq]
        else:
            freq = 1.0
    except (IndexError, AttributeError):
        freq = 1.0

    try:
        yearly_salary = round(amount * exchange_rates[currency] * freq, 2)
        return yearly_salary
    except KeyError:
        return np.nan


def batch_parse_salary(
    salary_strings: Iterable, exchange_rates: dict = None
) -> np.array:
    """

    Args:
        salary_strings (Iterable): _description_
        exchange_rates (dict, optional): _description_. Defaults to None.

    Returns:
        np.array: _description_
    """
    if not exchange_rates:
        print("Getting exhange rates...")
        exchange_rates = _get_conversion_rates(base_currency="GBP")
    parsed_salaries = np.empty_like(salary_strings, dtype="float")
    for i, salary in enumerate(salary_strings):
        parsed_salary = parse_salary(salary, exchange_rates)
        parsed_salaries[i] = parsed_salary
    return parsed_salaries


def combine_time_with_employer(years: int, months: int) -> int:
    """Combines time with employer (years, months) into total months.

    Args:
        years (int): Time spent with employer in years.
        months (int): Time spent with employer in months.

    Returns:
        int: Total months spent with employer.
    """
    return (years * 12) + months


def _pays_capital_tax(row: pd.Series) -> int:
    """Checks if customer pays capital tax and returns 1 if yes, 0 if no.

    Args:
        row (pandas.Series): Customer row from Mortgage table.

    Returns:
        int: 1 if pays capital tax, 0 if not.
    """
    if row["capital_gain"] == 0 and row["capital_loss"] == 0:
        return 0
    return 1


def import_campaign_table(path: str, self_learning_setup: bool = False) -> pd.DataFrame:
    """Imports relevant columns from Campaign table with correct dtypes and optionally drops
    rows with no response."""
    dtypes = {
        "age": int,
        "marital_status": "category",
        "job_title": "category",
        "occupation_level": int,
        "education_num": int,
        "familiarity_FB": int,
        "view_FB": int,
        "interested_insurance": int,
        "created_account": "string",
    }
    keep_cols = [
        "age",
        "marital_status",
        "occupation_level",
        "education_num",
        "familiarity_FB",
        "view_FB",
        "interested_insurance",
        "created_account",
    ]
    df = pd.read_csv(path, usecols=keep_cols, dtype=dtypes)
    if self_learning_setup:
        df["created_account"] = (
            df["created_account"].replace({"Yes": "1", "No": "0", np.nan: "-1"})
        ).astype("int")
    return df


def import_mortgage_table(path: str) -> pd.DataFrame:
    """Imports relevant columns from Mortgage table with correct dtypes.
    Also parses salary and total time with employer in months. Fills in '?' in workclass column."""
    dtypes = {
        "salary_band": "string",
        "years_with_employer": int,
        "months_with_employer": int,
        "hours_per_week": int,
        "capital_gain": int,
        "capital_loss": int,
        "workclass": "category",
    }
    keep_cols = [
        "salary_band",
        "years_with_employer",
        "months_with_employer",
        "hours_per_week",
        "capital_gain",
        "capital_loss",
        "workclass",
    ]
    df = pd.read_csv(path, usecols=keep_cols, dtype=dtypes)
    df["salary_band"] = batch_parse_salary(df["salary_band"])
    df["total_months_with_employer"] = df.apply(
        lambda x: combine_time_with_employer(
            x["years_with_employer"], x["months_with_employer"]
        ),
        axis=1,
    )
    df["pays_captial_tax"] = df.apply(_pays_capital_tax, axis=1)
    df["workclass"] = df["workclass"].replace({"?": np.nan})
    df.drop(
        [
            "years_with_employer",
            "months_with_employer",
            "capital_gain",
            "capital_loss",
        ],
        axis=1,
        inplace=True,
    )
    return df
