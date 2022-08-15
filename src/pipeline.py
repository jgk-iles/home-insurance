"""Function for constructing model Pipeline.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler


# Instantiate a RandomOverSampler
random_over_sampler = RandomOverSampler()

# Use column transformer to one-hot-encode only the categorical columns
column_transformer = ColumnTransformer(
    [("one_hot_encode", OneHotEncoder(), ["marital_status", "workclass"])],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

# Instantiate RandomForestClassifier
random_forest_classifier = RandomForestClassifier()


def make_pipeline() -> Pipeline:
    """Returns Pipeline used for Home Insurance Model

    Returns:
        imblearn.pipeline.Pipeline: Pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("random_over_sampler", random_over_sampler),
            ("encoder", column_transformer),
            ("random_forest", random_forest_classifier),
        ]
    )
    return pipeline
