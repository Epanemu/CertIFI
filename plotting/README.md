# Plotting scripts

**These are the scripts used to produce the figures in the paper.** They are provided as-is for
transparency. They do not run experiments. They parse the `output.txt` files (and, for the solver
progress figure, the SLURM logs) of experiments you have run yourself with
`experiment_eval.py` / `experiment_train.py` (see the main README).

The scripts are configured by module-level variables (input directory, number of seeds, directory
suffix, subplot grid), which you need to edit for each figure, as described below. Run them from the
repository root. Most of them write a PDF into `plots/` (create it first with `mkdir plots`), and
`final_eval_plot.py`, `final_train_plot.py` and `final_train_plot_dnfs.py` need the argument `plot`
(without it they only print summaries):

```bash
python plotting/final_eval_plot.py plot
```

Extra dependency: `seaborn` (for `final_train_plot_sparsity.py`).

## Expected directory layout

With `+dir_structure=cluster_batch_eval` / `+dir_structure=cluster_batch_train` the experiments are
stored as

```
batch_out/eval/experiment_eval/<method>-<scenario>/<seed>/output.txt
batch_out/train/experiment_train/<model>-<scenario>-<gamma>/<seed>/output.txt
batch_out/train/experiment_train/<model>-<scenario>-<n_terms>/<seed>/output.txt   # +dir_structure=cluster_batch_train_dnf
```

Point `base_dir_prefix` (or `base_dir_dense` / `base_dir_sparse`, `LOG_FILES`) to the folder that
contains the `<method>-<scenario>...` directories. Use a separate output folder for each setting,
e.g. by also overriding `hydra.sweep.dir`.

## Figures

| Figure (paper) | Script | What to set |
|---|---|---|
| Bias detection (Fig. 2), 10,000 s on 50,000 samples | `final_eval_plot.py` | `base_dir_prefix`; loop over 5 seeds (`range(5)` in `extract_data`) |
| Bias detection with 5 / 10 min limits (Fig. 5, 6) | `final_eval_plot.py` | as above, plus enable the `"Time [s]"` row (commented out in `extract_data`), `r, c = 2, 5` and `figsize=(12, 5)` |
| Bias detection on the full datasets (Fig. 7) | `final_eval_plot.py` | `base_dir_prefix`; single seed (`range(1)`) |
| Solver progress (Fig. 8) | `plot_mio_progress.py` | `LOG_FILES`: the solver logs (SLURM `*_log.out`) of the full-data runs with linear subgroups |
| Trained classifiers (Fig. 3) | `final_train_plot.py` | `base_dir_prefix` (dense runs), directory suffix `-0.01` and `range(5)` in `extract_data`; the 6 methods without `*_msd_*`/`*_spsf_*` in `method_order`; 3 rows (accuracy, FPSF train, FPSF test) |
| Detailed results, dense / sparse (Fig. 9, 10) | `final_train_plot.py` | as above with all methods in `method_order` and all 6 rows (`r, c = 6, 5`); `base_dir_prefix` pointing to the dense (`sparsity_term=0`) or sparse (`sparsity_term=0.1`) runs |
| Effect of sparsity (Fig. 4) | `final_train_plot_sparsity.py` | `base_dir_dense`, `base_dir_sparse` |
| DNF classifiers (Appendix C.3) | `final_train_plot_dnfs.py` | `base_dir_prefix`; directory suffix `-1` or `-3` (number of terms) in `extract_data` |

`final_train_plot.py` also prints LaTeX tables with the number of generated cuts (Table 1).
