"""Ablation driver for LinearFairClassifierFullRules (see linear_fair_classifier_full_rules.py).

Mirrors the "linear" branch of experiment_train.py in the repository root (data loading,
training, and the train/test accuracy + FPSF/MSD/SPSF bias evaluation block), but trains with
LinearFairClassifierFullRules instead of LinearFairClassifier: instead of lazily generating
fairness cuts via a Gurobi callback, every rule (conjunction) subgroup is enumerated upfront
and added as a regular constraint before the single solve.

Small smoke test:
    python ablations/full_rules/train_full_rules.py max_samples=200 time_limit=30

Setting used for the ablation:
    python ablations/full_rules/train_full_rules.py -m
"""
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bias_detect_mio import BiasDetect  # noqa: E402
from gerryfair.model import Auditor  # noqa: E402
from linear_fair_classifier_full_rules import (  # noqa: E402
    LinearFairClassifierFullRules,
    SubgroupBudgetExceeded,
)
from scenarios.folktables_scenarios import load_classif_scenario  # noqa: E402
from utils import accuracy, eval_fpsf, eval_msd, eval_spsf, f1_score  # noqa: E402


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig):
    binarizer, dhandler, X_orig, y_orig, X_orig_test, y_orig_test, binarizer_protected, dhandler_protected, protected_cols = (
        load_classif_scenario(cfg.scenario, cfg.seed, cfg.max_samples)
    )

    if cfg.scenario == "ACSMobility":
        cfg.gamma = cfg.gamma / 2

    X_prot_orig = X_orig[protected_cols]
    X_prot_orig_test = X_orig_test[protected_cols]

    n_samples = X_orig.shape[0]
    n_test = X_orig_test.shape[0]
    X_enc = dhandler.encode(X_orig)
    X_test_enc = dhandler.encode(X_orig_test)
    X_prot = binarizer_protected.encode(X_prot_orig, include_negations=False)
    X_prot_test = binarizer_protected.encode(X_prot_orig_test, include_negations=False)
    y = binarizer.encode_y(y_orig)
    y_test = binarizer.encode_y(y_orig_test)

    dfX_prot = pd.DataFrame(X_prot)
    dfX_prot_test = pd.DataFrame(X_prot_test)

    d_prot = X_prot_orig.shape[1]
    d_ext_prot = X_prot.shape[1]

    bias_type = cfg.bias_type.upper() if cfg.bias_type else ""

    t_start = time.time()
    mio_setup = LinearFairClassifierFullRules(
        gamma=(cfg.gamma * 4) if bias_type == "MSD" else cfg.gamma,
        fairness_type=bias_type,
        subgroup_type="rule",
        sparsity_term=cfg.sparsity_term,
    )
    enum_failed = False
    try:
        coefs, threshold = mio_setup.find_classifier(
            X_enc,
            X_prot,
            y,
            time_limit=cfg.time_limit,
            epsilon=1e-3,
            verbose=True,
            warmstart=True,
            n_min_group=cfg.n_min_group,
            max_subgroups=cfg.max_subgroups,
            solver_name=cfg.solver_name,
        )
    except SubgroupBudgetExceeded as e:
        enum_failed = True
        error_msg = str(e)
        coefs, threshold = None, None
    t = time.time()

    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_path = os.path.join(run_dir, "output.txt")

    if enum_failed:
        with open(out_path, "w") as out_file:
            out_file.write(f"Config:\n {cfg}\n")
            out_file.write("RESULT\n")
            out_file.write("Status: SUBGROUP ENUMERATION FAILED (budget exceeded)\n")
            out_file.write(f"Error: {error_msg}\n")
            out_file.write(f"Protected dimension: {d_prot} \n")
            out_file.write(f"Protected binarized dimension: {d_ext_prot} \n")
        print(f"Result saved to {out_path}")
        return

    y_hat_train = (X_enc @ coefs.reshape((-1, 1)) >= threshold).flatten()
    y_hat_train_prob = y_hat_train.astype(int)
    y_hat_test = (X_test_enc @ coefs.reshape((-1, 1)) >= threshold).flatten()
    y_hat_test_prob = y_hat_test.astype(int)

    print("TRAINED")

    mio = BiasDetect()
    _, ingroup = mio.find_subgroup_with_ingroup(X_prot, y_hat_train, time_limit=cfg.time_limit // 6, bias_type="msd", group_type="rule")
    msd_val_train = eval_msd(y_hat_train, ingroup)
    _, ingroup = mio.find_subgroup_with_ingroup(X_prot_test, y_hat_test, time_limit=cfg.time_limit // 6, bias_type="msd", group_type="rule")
    msd_val_test = eval_msd(y_hat_test, ingroup)

    auditor = Auditor(dfX_prot, np.zeros_like(y_orig), "FP")
    ingroup, oracle = auditor.audit(y_hat_train_prob, with_group_def=True)
    ingroup = np.array(ingroup, dtype=bool)
    spsf_val_train = eval_spsf(y_hat_train, ingroup)
    auditor = Auditor(dfX_prot_test, np.zeros_like(y_hat_test), "FP")
    ingroup, oracle = auditor.audit(y_hat_test_prob, with_group_def=True)
    ingroup = np.array(ingroup, dtype=bool)
    spsf_val_test = eval_spsf(y_hat_test, ingroup)

    auditor = Auditor(dfX_prot, y_orig.values, "FP")
    ingroup, oracle = auditor.audit(y_hat_train_prob, with_group_def=True)
    ingroup = np.array(ingroup, dtype=bool)
    fpsf_val_train = eval_fpsf(y_orig.values, y_hat_train, ingroup)
    auditor = Auditor(dfX_prot_test, y_orig_test.values, "FP")
    ingroup, oracle = auditor.audit(y_hat_test_prob, with_group_def=True)
    ingroup = np.array(ingroup, dtype=bool)
    fpsf_val_test = eval_fpsf(y_orig_test.values, y_hat_test, ingroup)

    _, ingroup = mio.find_subgroup_with_ingroup(X_prot, y_hat_train, time_limit=cfg.time_limit // 6, bias_type="spsf", group_type="rule")
    spsf_mio_val_train = eval_spsf(y_hat_train, ingroup)
    _, ingroup = mio.find_subgroup_with_ingroup(X_prot_test, y_hat_test, time_limit=cfg.time_limit // 6, bias_type="spsf", group_type="rule")
    spsf_mio_val_test = eval_spsf(y_hat_test, ingroup)

    mask = ~y
    fpsf_train_group_rule = mio.find_subgroup(X_prot[mask], y_hat_train[mask], time_limit=cfg.time_limit // 6, bias_type="spsf", group_type="rule")
    ingroup = np.ones_like(y, dtype=bool)
    for feat_i in fpsf_train_group_rule:
        ingroup &= X_prot[:, feat_i]
    fpsf_mio_val_train = eval_fpsf(y, y_hat_train, ingroup)
    mask_test = ~y_test
    fpsf_test_group_rule = mio.find_subgroup(X_prot_test[mask_test], y_hat_test[mask_test], time_limit=cfg.time_limit // 6, bias_type="spsf", group_type="rule")
    ingroup = np.ones_like(y_orig_test, dtype=bool)
    for feat_i in fpsf_test_group_rule:
        ingroup &= X_prot_test[:, feat_i]
    fpsf_mio_val_test = eval_fpsf(y_test, y_hat_test, ingroup)

    with open(out_path, "w") as out_file:
        out_file.write(f"Config:\n {cfg}\n")
        out_file.write("RESULT\n")
        out_file.write(f"Classif Coefs: {list(coefs)} \n")
        out_file.write(f"Classif Threshold: {threshold} \n")
        out_file.write(f"Status: {mio_setup.mio_result.solver.status} \n")
        out_file.write(f"Termination Condition: {mio_setup.mio_result.solver.termination_condition} \n")
        out_file.write(f"Number of rule subgroups enumerated: {mio_setup.n_subgroups} \n")
        out_file.write(f"Number of fairness constraints added: {mio_setup.n_cuts} \n")
        out_file.write(f"Enumeration time: {mio_setup.enum_time} \n")
        out_file.write(f"Train Accuracy: {accuracy(y, y_hat_train)} \n")
        out_file.write(f"Test Accuracy: {accuracy(y_test, y_hat_test)} \n")
        out_file.write(f"Train F1: {f1_score(y, y_hat_train)} \n")
        out_file.write(f"Test F1: {f1_score(y_test, y_hat_test)} \n")
        out_file.write(f"MSD: {msd_val_train} {msd_val_test} \n")
        out_file.write(f"Gerry FPSF: {fpsf_val_train} {fpsf_val_test} \n")
        out_file.write(f"Gerry SPSF: {spsf_val_train} {spsf_val_test} \n")
        out_file.write(f"MIO FPSF: {fpsf_mio_val_train} {fpsf_mio_val_test} \n")
        out_file.write(f"MIO SPSF: {spsf_mio_val_train} {spsf_mio_val_test} \n")
        out_file.write(f"Time: {t - t_start} \n")
        out_file.write(f"True number of training samples: {n_samples} \n")
        out_file.write(f"True number of testing samples: {n_test} \n")
        out_file.write(f"Protected dimension: {d_prot} \n")
        out_file.write(f"Protected binarized dimension: {d_ext_prot} \n")

    print(f"Result saved to {out_path}")
    print(f"Train Accuracy: {accuracy(y, y_hat_train):.4f}  Test Accuracy: {accuracy(y_test, y_hat_test):.4f}")
    print(f"Gerry FPSF  train/test: {fpsf_val_train:.4f} / {fpsf_val_test:.4f}")
    print(f"MIO   FPSF  train/test: {fpsf_mio_val_train:.4f} / {fpsf_mio_val_test:.4f}")
    print(f"Gerry SPSF  train/test: {spsf_val_train:.4f} / {spsf_val_test:.4f}")
    print(f"MIO   SPSF  train/test: {spsf_mio_val_train:.4f} / {spsf_mio_val_test:.4f}")
    print(f"MSD         train/test: {msd_val_train:.4f} / {msd_val_test:.4f}")


if __name__ == "__main__":
    run_experiment()
