# Ablation: all fairness constraints upfront

**This code was used only for an ablation study. It is not part of the main method.**

The main training framework (`experiment_train.py`, `linear_fair_classifier.py`) adds fairness
constraints *lazily*: whenever the solver finds a new incumbent, the most violating subgroup is
searched for and, if it violates the threshold, added as a cut.

This ablation instead enumerates *every* conjunction subgroup of the protected attributes upfront
(pruning only empty subgroups, or those smaller than `n_min_group`) and adds a fairness constraint
for each of them before a single solve (the "extended formulation" in Appendix B of the paper).
Only conjunction (rule) subgroups are supported.

- `linear_fair_classifier_full_rules.py`: the `LinearFairClassifierFullRules` model. It uses the
  same linear-classifier formulation as `linear_fair_classifier.py`, and needs no solver
  callbacks, so training can use any Pyomo-supported MIO solver (`solver_name`, e.g. `appsi_highs`).
  The post-training bias evaluation still uses Gurobi.
- `train_full_rules.py`: the experiment driver, a Hydra app configured by `conf/config.yaml`.
  Outputs have the same format as `experiment_train.py` (`output.txt` in the Hydra run directory).

Run from the repository root:

```bash
# small smoke test
python ablations/full_rules/train_full_rules.py max_samples=200 time_limit=30

# the setting used in the ablation (5 seeds x 2 measures x 5 datasets)
python ablations/full_rules/train_full_rules.py -m
```

For datasets with many protected attributes, the number of conjunction subgroups goes into the
millions (see Table 1 of the paper). In that case the enumeration can be very slow or run out of
memory; set `max_subgroups` to fail early.
