"""Tests for salary parser.
"""

import numpy as np
from src.preprocessing import batch_parse_salary, parse_salary

TEST_CONVERSIONS = {
    "GBP": 1.0,
    "GTQ": 0.11,
    "SDG": 0.0015,
    "GMD": 0.015,
    "DZD": 0.0057,
    "AOA": 0.0019,
}

TEST_INPUT = [
    "£16486 yearly",
    "£733.14 pw",
    "£20042 yearly",
    "44462895188.2GTQ",
    "1675262.85SDG",
    "£27326 yearly",
    "£403.48 pw",
    "£325.06 pw",
    "£46653 yearly",
    "£22028 yearly",
    "£22092 yearly",
    "416544.02GMD",
    "362461282142.49DZD",
    "£26851 - 48668 range",
    "£293.36 pw",
    "£2120.82 per month",
    "£1549.02 per month",
    "£39111 yearly",
    "£1592.17 per month",
    "£17178 yearly",
    "£17747 yearly",
    "£339.61 pw",
    "£421.7 pw",
    "£15180 - 27514 range",
    "£21670 - 39277 range",
    "94860480.06AOA",
    "£17575 yearly",
    "£17701 - 32084 range",
    "£428.88 pw",
    "£14536 - 26347 range",
    "AUD",
]

TEST_EXPECTED = np.array(
    [
        16486.0,
        38123.28,
        20042.0,
        4890918470.7,
        2512.89,
        27326.0,
        20980.96,
        16903.12,
        46653.0,
        22028.0,
        22092.0,
        6248.16,
        2066029308.21,
        37759.5,
        15254.72,
        25449.84,
        18588.24,
        39111,
        19106.04,
        17178.0,
        17747.0,
        17659.72,
        21928.4,
        21347.0,
        30473.5,
        180234.91,
        17575.0,
        24892.5,
        22301.76,
        20441.5,
        np.nan,
    ]
)


def test__parse_salary():
    """Test for `parse_salary()` function."""
    for salary, expected in zip(TEST_INPUT, TEST_EXPECTED):
        parsed_salary = parse_salary(salary, TEST_CONVERSIONS)
        if np.isnan(parsed_salary):
            assert parsed_salary != expected
        else:
            assert parsed_salary == expected


def test_batch_parse_salary():
    """Test for `batch_parse_salary` function."""
    parsed_salaries = batch_parse_salary(TEST_INPUT, TEST_CONVERSIONS)
    np.testing.assert_equal(parsed_salaries, TEST_EXPECTED)
