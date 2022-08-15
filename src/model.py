"""Script for building the model.

Campaign and Mortgage tables should be present in "../data/" in csv format.
"""

import sys
from pathlib import Path
import pandas as pd
import joblib

# Append src to PYTHONPATH
ROOT_PATH = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT_PATH / "src"
sys.path.append(SRC_PATH)

from pipeline import make_pipeline
from preprocessing import import_campaign_table, import_mortgage_table

# Relative paths for data files
ROOT_PATH = Path(__file__).resolve().parent.parent
CAMPAIGN_TABLE_PATH = ROOT_PATH / "data/Campaign.csv"
MORTGAGE_TABLE_PATH = ROOT_PATH / "data/Mortgage.csv"

MODEL_FEATURES = [
    "age",
    "marital_status",
    "occupation_level",
    "education_num",
    "familiarity_FB",
    "view_FB",
    "interested_insurance",
    "salary_band",
    "hours_per_week",
    "workclass",
    "total_months_with_employer",
    "pays_captial_tax",
]


if __name__ == "__main__":
    campaign_df = import_campaign_table(CAMPAIGN_TABLE_PATH)
    mortgage_df = import_mortgage_table(MORTGAGE_TABLE_PATH)

    # Concat the tables
    combined_df = pd.concat([campaign_df, mortgage_df], axis=1)
    combined_df = combined_df.dropna()

    # X, y split for labelled data
    labelled = combined_df[combined_df.created_account != -1]
    X_lab, y_lab = (
        labelled[MODEL_FEATURES],
        labelled["created_account"],
    )

    # Build Pipeline
    pipeline = make_pipeline()

    # Fit Pipeline
    pipeline.fit(X_lab, y_lab)

    # Save the model as a pickle file
    SAVE_PATH = ROOT_PATH / "assets" / "pipeline.pkl"
    joblib.dump(pipeline, SAVE_PATH)
