# CertIFI: Certifiably Intersectionally Fair & Interpretable models

Code for the paper **Intersectional Fairness via Mixed-Integer Optimization**
by Jiří Němeček, Mark Kozdoba, Illia Kryvoviaz, Tomáš Pevný and Jakub Mareček (NeurIPS 2026).

The repository contains a Mixed-Integer Optimization (MIO) framework for

1. **detecting the most biased subgroup** in data or in the predictions of a classifier, under
   the Statistical Parity Subgroup Fairness (SPSF) and Subgroup Discrepancy (SD / MSD) measures, with
   subgroups described either as conjunctions of protected attribute values or as linear threshold
   functions of the protected attributes, and
2. **training interpretable classifiers** (linear models and DNFs) **under intersectional fairness
   constraints** (FPSF, SPSF, MSD), where the exponentially many fairness constraints are added
   lazily as cutting planes, using the detection MIO to find the most violating subgroup.

All formulations are written in [Pyomo](https://www.pyomo.org/) and solved with
[Gurobi](https://www.gurobi.com/).

## Installation

Tested with Python 3.11.

```bash
pip install -r requirements.txt
```

**Solver.** The experiments in the paper use Gurobi, which requires a license
([free for academic use](https://www.gurobi.com/academia/academic-program-and-licenses/)).
The size-limited license bundled with `pip install gurobipy` is not sufficient for realistic data sizes.
Training uses Gurobi-specific lazy-constraint callbacks (`gurobi_persistent`), so it requires Gurobi.
Bias detection (`experiment_eval.py` with `solver=highs`) can also use the open-source
[HiGHS](https://highs.dev/) solver (`pip install highspy`).

**Data.** The datasets come from the US Census (ACS PUMS, 2018, 1-Year, California) via
[folktables](https://github.com/socialfoundations/folktables). They are downloaded automatically
into `./data/` on first use (about 250 MB).

## Repository structure

| Path | Content |
|---|---|
| `experiment_eval.py` | Bias detection experiments: finding the most biased subgroup in the data (Sec. 4.1, App. C.1) |
| `experiment_train.py` | Training experiments: fair linear / DNF classifiers and baselines (Sec. 4.2, App. C.2, C.3) |
| `bias_detect_mio.py` | `BiasDetect`: MIO formulations for finding the most biased subgroup (SD/MSD and SPSF objectives; conjunction and linear subgroups) |
| `linear_fair_classifier.py` | `LinearFairClassifier`: MIO training of a (optionally sparse) linear classifier with lazily generated fairness constraints |
| `dnf_fair_classifier.py` | `DNFFairClassifier`: the same for DNF (rule set) classifiers |
| `utils.py` | Evaluation of the fairness measures (MSD, SPSF, FPSF) for a given subgroup, accuracy, F1 |
| `scenarios/` | Loading and subsampling of the folktables datasets, definition of protected attributes |
| `data_handler/`, `binarizer/` | Data encoding (one-hot, scaling) and binarization of features |
| `gerryfair/` | Modified copy of [GerryFair](https://github.com/algowatchPenn/GerryFair) (MIT), used as a baseline; see `gerryfair/MODIFICATIONS.md` |
| `conf/` | [Hydra](https://hydra.cc/) configuration of the experiments |
| `ablations/full_rules/` | Ablation only: all conjunction-subgroup constraints added upfront instead of lazily (App. B) |
| `plotting/` | Scripts used to produce the figures from experiment outputs |

## Running experiments

Both experiment scripts are Hydra apps. Settings are in `conf/config_eval.yaml` and
`conf/config_train.yaml`, and the defaults match the paper. Any key can be overridden on the
command line. A single run writes its results to `outputs/<date>/<time>/output.txt`. A sweep (`-m`)
runs over the parameters in the `hydra.sweeper.params` section of the config and writes to
`multirun/...`, or with `+dir_structure=cluster_batch_eval` / `+dir_structure=cluster_batch_train`
to `batch_out/<eval|train>/<script>/<method>-<scenario>[-<gamma>]/<seed>/`.

Datasets (`scenario`): `ACSIncome`, `ACSPublicCoverage`, `ACSMobility`, `ACSEmployment`,
`ACSTravelTime`. Every configuration in the paper is run with seeds 0 to 4, which determine the
random data subsample.

To run a sweep on a SLURM cluster, install `hydra-submitit-launcher` and add
`hydra/launcher=slurm hydra.launcher.partition=<partition>` (see `conf/hydra/launcher/slurm.yaml`).

### Bias detection (`experiment_eval.py`)

`method` is `mio_<objective>_<subgroups>` or `gerry_spsf`:

- objective `msd` maximizes Subgroup Discrepancy (for conjunction subgroups, this is MSD); `spsf` maximizes SPSF,
- subgroups `rule` are conjunctions of protected attribute values, `linear` are linear threshold functions,
- `gerry_spsf` is the GerryFair auditor (SPSF, linear subgroups).

Linear subgroups are warm-started with the GerryFair solution if it is numerically usable,
otherwise with a logistic regression model.

```bash
# single run
python experiment_eval.py method=mio_msd_rule scenario=ACSIncome seed=0

# Figure 2: 50,000 samples, 10,000 s limit, all methods, datasets and seeds
python experiment_eval.py -m +dir_structure=cluster_batch_eval

# Appendix C.1: shorter time limits, and the full datasets with a 24 h limit
python experiment_eval.py -m +dir_structure=cluster_batch_eval time_limit=300
python experiment_eval.py -m +dir_structure=cluster_batch_eval time_limit=600
python experiment_eval.py -m +dir_structure=cluster_batch_eval time_limit=86400 max_samples=1000000 seed=0
```

(Use a different output directory for each setting, e.g. via `hydra.sweep.dir=...`.)

`output.txt` contains the configuration, the found subgroup (the conjunction, or the coefficients and
threshold of the linear function), its `SPSF` and `MSD` values, the `Time` taken, and whether the
solution was proven `Optimal`.

### Training (`experiment_train.py`)

`model` is `<predictor>[_<measure>_<subgroups>]`:

| Part | Values |
|---|---|
| predictor | `linear` (linear classifier), `dnf` (DNF with `n_terms` terms), `gerry` (GerryFair baseline) |
| measure | `fpsf`: FPSF constraints, violating subgroup found by maximizing FPSF <br> `msd0`: FPSF constraints, violating subgroup found by maximizing SD on the negative samples ("FPSF by SD") <br> `spsf`: SPSF constraints <br> `msd`: MSD constraints (with threshold 4γ) |
| subgroups | `rule` (conjunctions, S_A) or `linear` (linear threshold functions, S_L) |

`linear` and `dnf` without a suffix are the models trained without fairness constraints. GerryFair
supports `gerry_fpsf` and `gerry_spsf`.

Paper settings, which are the config defaults: 10,000 training and 10,000 test samples, a 4 h time limit, and 5 min to
find a violating subgroup. The fairness threshold is γ = 0.01, halved automatically for ACSMobility. GerryFair runs
`time_limit / 5` iterations.

```bash
# single run
python experiment_train.py model=linear_fpsf_rule scenario=ACSIncome seed=0

# Figure 3 (dense linear models): base model, 4 fair MIO models and GerryFair
python experiment_train.py -m +dir_structure=cluster_batch_train

# Appendix C.2 (Figure 9): additionally SPSF and MSD constraints
python experiment_train.py -m +dir_structure=cluster_batch_train \
    model=linear_spsf_rule,linear_msd_rule,linear_spsf_linear,linear_msd_linear

# Sparse linear models (Figure 4, Figure 10)
python experiment_train.py -m +dir_structure=cluster_batch_train sparsity_term=0.1 \
    model=linear,linear_fpsf_rule,linear_msd0_rule,linear_fpsf_linear,linear_msd0_linear,linear_spsf_rule,linear_msd_rule,linear_spsf_linear,linear_msd_linear

# DNF classifiers (Appendix C.3)
python experiment_train.py -m +dir_structure=cluster_batch_train_dnf \
    model=dnf,dnf_fpsf_rule,dnf_msd0_rule n_terms=1,3
```

(Again, use separate output directories for the dense and sparse runs.)

After training, the classifier is audited on the training and test sets with GerryFair and with
MIO over conjunction subgroups. `output.txt` contains the learned model, the solver status, the
number of generated cuts (fairness constraints), train/test accuracy and F1, and the train/test
values of MSD, FPSF and SPSF found by each auditor (`Gerry FPSF`, `MIO FPSF`, ...).

### Using the components directly

```python
from bias_detect_mio import BiasDetect
from linear_fair_classifier import LinearFairClassifier

# X_prot: binarized protected attributes (bool), y: labels or predictions (bool)
rule = BiasDetect().find_subgroup(X_prot, y, bias_type="msd", group_type="rule", time_limit=300)

# X: encoded features (float)
clf = LinearFairClassifier(gamma=0.01, fairness_type="FPSF", subgroup_type="rule")
coefs, threshold = clf.find_classifier(X, X_prot, y, time_limit=3600, epsilon=1e-3, warmstart=True)
```

See `experiment_eval.py` and `experiment_train.py` for how the data is encoded.

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

The code is released under the [CC BY 4.0](LICENSE) license. The `gerryfair/` directory is a modified
copy of GerryFair, distributed under its original MIT license (`gerryfair/LICENSE`).
