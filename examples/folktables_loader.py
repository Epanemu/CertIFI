"""Loading of the US Census (ACS PUMS) datasets from folktables
(https://github.com/socialfoundations/folktables), as used in the paper.

The data are downloaded into ./data on first use. Requires `pip install folktables`.
"""

import numpy as np
import pandas as pd
from folktables import ACSDataSource

DATASETS = ["ACSIncome", "ACSPublicCoverage", "ACSMobility", "ACSEmployment", "ACSTravelTime"]

# https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.pdf
PROTECTED_ATTRS = [
    "SEX", "RAC1P", "AGEP", "POBP", "DIS", "CIT", "MIL", "ANC", "NATIVITY",
    "DEAR", "DEYE", "DREM", "FER", "POVPIP",
]
CONTINUOUS_FEATURES = ["AGEP", "PINCP", "WKHP", "JWMNP", "POVPIP"]

# simplify features with too many values
FEATURE_PROCESSING = {
    "POBP": lambda x: int(x) // 100,  # place of birth: country -> continent
    "OCCP": lambda x: int(x) // 100,
    "PUMA": lambda x: int(x) // 100,
    "POWPUMA": lambda x: int(x) // 1000,
}

# readable values of some protected attributes (shortened)
VALUE_NAMES = {
    "SEX": {1: "Male", 2: "Female"},
    "RAC1P": {
        1: "White", 2: "Black", 3: "American Indian", 4: "Alaska Native",
        5: "Native tribes specified", 6: "Asian", 7: "Pacific Islander",
        8: "Some Other Race", 9: "Two or More Races",
    },
    "POBP": {0: "US", 1: "Europe", 2: "Asia", 3: "Non-US Americas", 4: "Africa", 5: "Oceania"},
    "DIS": {1: "With a disability", 2: "Without a disability"},
    "NATIVITY": {1: "Native", 2: "Foreign born"},
}


def load_dataset(name: str, n_samples: int, seed: int = 0, state: str = "CA", year: str = "2018"):
    """Load a folktables dataset and draw a random train/test split of `n_samples` each.

    Returns:
        (X_train, y_train, P_train), (X_test, y_test, P_test): features (all, including
        protected), boolean labels and protected attributes, as DataFrames / arrays.
        Categorical features are converted to strings, continuous stay numeric.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name}, choose one of {DATASETS}")
    import folktables

    task = getattr(folktables, name)
    data = ACSDataSource(survey_year=year, horizon="1-Year", survey="person").get_data(states=[state], download=True)
    X, y, _ = task.df_to_pandas(data)
    mask = ~X.isnull().any(axis=1)
    X, y = X[mask.values].copy(), y[mask.values].iloc[:, 0].to_numpy().astype(bool)

    for col in list(X.columns):
        if col in FEATURE_PROCESSING:
            X[col] = X[col].map(FEATURE_PROCESSING[col])
        if X[col].nunique() <= 1:
            X = X.drop(columns=[col])
        elif col not in CONTINUOUS_FEATURES:
            X[col] = X[col].astype(int).map(lambda v, c=col: VALUE_NAMES.get(c, {}).get(v, str(v)))

    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(2 * n_samples, X.shape[0]), replace=False)
    train, test = idx[: len(idx) // 2], idx[len(idx) // 2 :]
    protected = [c for c in X.columns if c in PROTECTED_ATTRS]

    def part(ix):
        Xp = X.iloc[ix].reset_index(drop=True)
        return Xp, y[ix], Xp[protected]

    return part(train), part(test)
