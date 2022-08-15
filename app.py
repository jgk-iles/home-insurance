"""Script to run Streamlit app.
"""

from datetime import datetime

import pickle
import joblib

import streamlit as st
import pandas as pd

from src.utils import diff_month

# Worked out from notebook
THRESHOLD = 0.36

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

# Open dictionary of job title scores
with open("assets/job_dict.pkl", "rb") as handle:
    job_dict = pickle.load(handle)

# Open dictionary of education_levels
with open("assets/edu_dict.pkl", "rb") as handle:
    edu_dict = pickle.load(handle)

# Title
st.header("Home Insurance Survey")

# Get age
age = st.number_input("What is your age?", min_value=0, max_value=100)

# Get marital status
marital_status = st.selectbox(
    "Please select your marital status.",
    options=[
        "Never-married",
        "Married-civ-spouse",
        "Divorced",
        "Married-spouse-absent",
        "Separated",
        "Married-AF-spouse",
        "Widowed",
    ],
)

# Get level of education
education = st.selectbox(
    "Please select your highest level of education", options=edu_dict.keys()
)

# Get workclass
workclass = st.selectbox(
    "Please select which working situation describes you best.",
    options=[
        "State-gov",
        "Self-emp-not-inc",
        "Private",
        "Federal-gov",
        "Local-gov",
        "Self-emp-inc",
        "Without-pay",
        "Never-worked",
    ],
)

# Get job title
job_title = st.selectbox(
    "Please select your job from the list.", options=job_dict.keys()
)

# Get salary
salary = st.number_input("Please input your yearly salary in Pounds (£).")

# Get hours worked per week
hours_per_week = st.number_input(
    "Please input the average numbers of hours per week that you work.", min_value=0
)

# Get date started working for employer
time_with_employer = st.date_input(
    "When did you start working for your current employer?",
    min_value=datetime(1900, 1, 1),
)

# Get familiarity with FB
familiarity_fb = st.slider(
    "How familiar are you with the FutureBank brand? (1 = don't know it, 10 = a big fan, follow on social media)",
    min_value=1,
    max_value=10,
)

# Get view of FB
view_fb = st.slider(
    "What is your view of FutureBank? (1 = really dislike, 10 = really like)",
    min_value=1,
    max_value=10,
)

# Ask if interested in insurance
interested_insurance = st.radio(
    "Are you interested in purchasing home insurance?", options=["Yes", "No"]
)
if interested_insurance == "Yes":
    interested_insurance = 1
else:
    interested_insurance = 0

# Ask if they pay capital tax
pays_capital_tax = st.radio(
    "Have you ever paid capital gains/loss tax?", options=("Yes", "No")
)
if pays_capital_tax == "Yes":
    pays_capital_tax = 1
else:
    pays_capital_tax = 0


if st.button("Submit"):

    survey_result = pd.DataFrame(
        [
            [
                age,
                marital_status,
                job_dict[job_title],
                edu_dict[education],
                familiarity_fb,
                view_fb,
                interested_insurance,
                salary,
                hours_per_week,
                workclass,
                diff_month(datetime.now(), time_with_employer),
                pays_capital_tax,
            ]
        ],
        columns=MODEL_FEATURES,
    )
    pipeline = joblib.load("assets/pipeline.pkl")
    prediction = pipeline.predict_proba(survey_result)[0][1]

    if prediction > THRESHOLD:
        st.success("Would you like to purchase home insurance?")
    else:
        st.success("Thank you for your submission")

    with st.expander("See explanation"):
        st.dataframe(survey_result)
        st.info(
            f"Our model gave you a prediction probability of {prediction:0.2f}. The threshold for offering home insurance is {THRESHOLD:0.2f}."
        )
