# CertIFI: Certifiably Intersectionally Fair & Interpretable models

`certifi-learn` finds the **most biased subgroup** in data or in a model's predictions,
and trains **interpretable classifiers** (linear models and rule sets) that are
**certifiably fair** on all intersectional subgroups, for example all combinations
of sex, race and age group.

> [!NOTE]
> This library extends the NeurIPS 2026 paper
> *Intersectional Fairness via Mixed-Integer Optimization* (see [Citation](#citation)).
> Notably, it adds an exact branch-and-bound subgroup search (implemented in Rust)
> that replaces the Mixed-Integer Optimization (MIO) search for conjunction
> subgroups by default. The code used for the experiments in the paper is available
> at the tag [`neurips2026`](https://github.com/Epanemu/CertIFI/tree/neurips2026).

## Features

- **Bias detection.** Find the subgroup with the largest violation of
  - Subgroup Discrepancy / MSD,
  - Statistical Parity Subgroup Fairness (SPSF),
  - False Positive Subgroup Fairness (FPSF).

  Subgroups can be
  - **conjunctions** of protected attribute values (e.g. `SEX = Female AND RAC1P = Black`). The exact search is fast, needs no solver, and its result is proven optimal.
  - **linear threshold functions** of the protected attributes, found by MIO.
- **Fair training.** Linear classifiers and DNFs (rule sets), trained to optimality
  under the constraint that *every* subgroup has bias at most γ. Fairness
  constraints are generated lazily: only the subgroups that are actually violated
  are added, so there's no need to enumerate the exponentially many subgroups.
- A scikit-learn-style API that works with pandas DataFrames. Categorical
  features are encoded automatically.

## Installation

```bash
pip install "certifi-learn[gurobi] @ git+https://github.com/Epanemu/CertIFI"
```

Building from source requires a [Rust toolchain](https://rustup.rs/).

Optional dependencies (extras):

| Extra | Needed for |
|---|---|
| `gurobi` | Training classifiers (Gurobi lazy-constraint callbacks) and MIO detection. Requires a [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/), free for academic use. |
| `highs` | MIO detection with the open-source [HiGHS](https://highs.dev/) solver, an alternative to Gurobi |
| `mio` | Pyomo only, if you bring your own solver installation |
| `examples` | The folktables example |

Bias detection on conjunction subgroups needs no extras.

## Quickstart

### Finding the most biased subgroup

```python
from certifi_learn import find_subgroup
from certifi_learn.encoding import TabularEncoder

# protected: DataFrame of protected attributes, y: binary labels, y_pred: model predictions
enc = TabularEncoder(protected)                 # binarizes into literals like "SEX = Female"
P = enc.binary(protected, negations=False)
names = enc.binary_names(negations=False)

# bias in the data: subgroup whose share differs most between positive and negative labels
sg = find_subgroup(P, y, measure="msd")
print(sg.value, sg.optimal, sg.describe(names))

# bias of a classifier
sg = find_subgroup(P, y_pred, measure="fpsf", y_true=y)

# linear subgroups (MIO, requires a solver)
sg = find_subgroup(P, y_pred, measure="spsf", subgroups="linear", solver="highs", time_limit=300)
```

`find_subgroup` returns a `DetectedSubgroup` with the violation `value`, the
membership `mask`, `optimal` (whether the subgroup is proven to be the most biased),
and `describe()` / `contains(X_new)`. If you pass `gamma=...`, it returns a subgroup
only if one violates the threshold, and `None` otherwise. That `None` certifies
γ-fairness, unless a warning reports that the search was stopped by the time limit.

### Training a fair classifier

```python
from certifi_learn import FairLinearClassifier, FairDNFClassifier

clf = FairLinearClassifier(gamma=0.01, measure="fpsf", time_limit=600)
clf.fit(X, y, X_protected)          # X may include the protected attributes
y_pred = clf.predict(X_test)

print(clf.n_cuts_)                   # number of subgroups constrained during training
print(clf.describe_cut_subgroups())  # which ones

rules = FairDNFClassifier(n_terms=3, gamma=0.01, measure="fpsf").fit(X, y, X_protected)
print(rules.describe())
```

Main parameters:

| Parameter | Values |
|---|---|
| `measure` | `"fpsf"`, `"fpsf_by_sd"` (FPSF constraints, with violating subgroups found by maximizing Subgroup Discrepancy on negative samples), `"spsf"`, `"msd"`, or `None` (no fairness constraint) |
| `subgroups` | `"conjunction"` or `"linear"` |
| `gamma` | Fairness threshold. Every subgroup must have bias ≤ γ on the training data. |
| `detection_backend` | `"exact"` (default) or `"mio"`, for finding violating conjunction subgroups. Linear subgroups always use MIO. |
| `sparsity` | (linear only) Penalty on the proportion of non-zero coefficients |
| `time_limit` | Solver time limit in seconds. The best solution found so far is returned. |

γ-fairness is guaranteed on the training data. On new data, the bias can be
somewhat higher. Audit your model on held-out data with `find_subgroup`.

## Bias measures

For a classifier h and a subgroup S:

| Measure | Definition |
|---|---|
| SPSF | P(S) · \|P(h=1) − P(h=1 \| S)\| |
| FPSF | P(S, y=0) · \|P(h=1 \| y=0) − P(h=1 \| S, y=0)\| |
| SD / MSD | \|P(S \| y=1) − P(S \| y=0)\|, i.e. how differently S is represented among positive and negative samples. Maximized over conjunctions, this is the Maximum Subgroup Discrepancy. |

`certifi_learn.measures` evaluates these for a given subgroup mask.
`certifi_learn.exact` exposes the Rust search directly (`msd`, `spsf`, `fpsf`), with
options such as sample weights, negated literals and grouped one-hot columns. See
[`docs/subgroup_pruning_proof.tex`](docs/subgroup_pruning_proof.tex) for the
algorithm and its proof of correctness.

## Example

[`examples/detect_and_train.py`](examples/detect_and_train.py) runs end to end on
the US Census data (ACS Income, via [folktables](https://github.com/socialfoundations/folktables)).
It finds the most biased subgroup in the data, trains an unconstrained and a fair
linear model, and audits both on test data:

```bash
pip install "certifi-learn[gurobi,examples] @ git+https://github.com/Epanemu/CertIFI"
python examples/detect_and_train.py
```

## Development

```bash
pip install -e ".[gurobi,highs,test,examples]"   # builds the Rust extension
pytest
```

Tests that need HiGHS or a Gurobi license are skipped if those are not available.

## Citation

```bibtex
@inproceedings{nemecek2026intersectional,
 author = {N\v{e}me\v{c}ek, Ji\v{r}\'{\i} and Kozdoba, Mark and Kryvoviaz, Illia and Pevn\'{y}, Tom\'{a}\v{s} and Mare\v{c}ek, Jakub},
 booktitle = {Advances in Neural Information Processing Systems},
 publisher = {Curran Associates, Inc.},
 title = {Intersectional Fairness via Mixed-Integer Optimization},
 volume = {39, Main Conference},
 year = {2026}
}
```

## License

[MIT](LICENSE)
