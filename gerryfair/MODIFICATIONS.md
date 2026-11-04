# Modifications to GerryFair

This directory contains a modified copy of the GerryFair package by Kearns, Neel, Roth and Wu
(https://github.com/algowatchPenn/GerryFair, MIT License, see `LICENSE`).
It is used as the baseline in our experiments. Results obtained with it may differ from
the upstream `pip install gerryfair` package due to the following changes:

- `model.py`
  - Added the `"SP"` (statistical parity) fairness option. It is implemented by auditing
    false positives with all labels set to 0, which turns FP auditing into SP (SPSF) auditing.
  - Added an optional `time_limit` (seconds, default `None` = no limit) that stops
    fictitious play after the given wall-clock time.
- `auditor.py`
  - `Auditor.audit(predictions, with_group_def=False)`: with `with_group_def=True`, it also
    returns the regression oracle that defines the found subgroup.
  - `get_group`: the false-positive rate of the negated group is computed over the members of
    the negated group (upstream uses the members of the original group).
- Code was reformatted (black).
