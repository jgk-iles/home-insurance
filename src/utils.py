from datetime import datetime


def diff_month(date_one: datetime, date_two: datetime) -> int:
    """Gets the difference in months between two dates.

    Args:
        d1 (datetime): Most recent date.
        d2 (datetime): Oldest date.

    Returns:
        int: Number of months between the two dates.
    """
    return (date_one.year - date_two.year) * 12 + date_one.month - date_two.month
