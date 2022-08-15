"""Script predicting new batches of customers. 
"""

import sys
from pathlib import Path
import pandas as pd
import joblib

# Append src to PYTHONPATH
ROOT_PATH = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT_PATH / "src"
sys.path.append(SRC_PATH)

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
    combined_df = combined_df.dropna(subset=MODEL_FEATURES)

    # Get X
    X = combined_df[MODEL_FEATURES]

    # Load model
    MODEL_PATH = ROOT_PATH / "assets/pipeline.pkl"
    pipeline = joblib.load(MODEL_PATH)

    # Predict probabilities
    predict_probas = pipeline.predict_proba(X)
    pd.DataFrame(predict_probas).to_csv(ROOT_PATH / "outputs/out.csv")
