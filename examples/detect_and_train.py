"""Example: find the most biased subgroup in data, then train a fair interpretable classifier.

Uses the ACS Income dataset (US Census, California 2018) from folktables:

    pip install "certifi-learn[gurobi,examples]"
    python examples/detect_and_train.py

Training requires a Gurobi license; the detection part runs without any solver.
"""

import numpy as np

from certifi_learn import FairLinearClassifier, find_subgroup
from certifi_learn.encoding import TabularEncoder
from folktables_loader import load_dataset

(X, y, P), (X_test, y_test, P_test) = load_dataset("ACSIncome", n_samples=2000, seed=0)
print(f"{X.shape[0]} training samples, protected attributes: {list(P.columns)}")

# --- 1. Bias in the data -------------------------------------------------------
# Binarize the protected attributes into literals such as "SEX = Female" or "AGEP between [17, 25)"
encoder = TabularEncoder(P)
P_bin = encoder.binary(P, negations=False)
names = encoder.binary_names(negations=False)

# Maximum Subgroup Discrepancy: the conjunction whose frequency differs most
# between people with positive and negative labels (exact search, no solver needed)
sg = find_subgroup(P_bin, y, measure="msd")
print(f"\nMost biased subgroup in the data (MSD = {sg.value:.3f}, proven optimal: {sg.optimal}):")
print("   ", sg.describe(names))

# --- 2. Training ----------------------------------------------------------------
# An unconstrained linear model, and one constrained to 0.01-FPSF fairness on all
# conjunctions of protected attributes
for gamma, measure in [(None, None), (0.01, "fpsf")]:
    clf = FairLinearClassifier(gamma=gamma or 0, measure=measure, time_limit=120)
    clf.fit(X, y, P)
    y_hat = clf.predict(X_test)

    # audit the trained model on test data: most violating subgroup w.r.t. FPSF
    P_test_bin = clf.protected_encoder_.binary(P_test, negations=False)
    worst = find_subgroup(P_test_bin, y_hat, measure="fpsf", y_true=y_test)

    label = "no fairness constraint" if measure is None else f"{gamma}-FPSF fair"
    print(f"\nLinear model, {label}:")
    print(f"    test accuracy {np.mean(y_hat == y_test):.3f}, test FPSF {worst.value:.4f}")
    print(f"    most unfair test subgroup: {worst.describe(clf.protected_feature_names_)}")
    if measure is not None:
        print(f"    {clf.n_cuts_} subgroups were constrained during training:")
        for d in clf.describe_cut_subgroups():
            print("       ", d)
