import os
import pickle
import subprocess
import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import time

from dnf_fair_classifier import DNFFairClassifier
from gerryfair.model import Auditor, Model
from linear_fair_classifier import LinearFairClassifier
from scenarios.folktables_scenarios import load_classif_scenario
from bias_detect_mio import BiasDetect
from utils import eval_fpsf, eval_spsf, eval_msd, accuracy, f1_score

gitcommit = ""


@hydra.main(version_base="1.3", config_path="conf", config_name="config_train")
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
    X = binarizer.encode(X_orig, include_negations=True)
    X_test =  binarizer.encode(X_orig_test, include_negations=True)
    X_prot = binarizer_protected.encode(X_prot_orig, include_negations=False)
    X_prot_test = binarizer_protected.encode(X_prot_orig_test, include_negations=False)
    y = binarizer.encode_y(y_orig)
    y_test = binarizer.encode_y(y_orig_test)

    X_enc = dhandler.encode(X_orig)
    X_test_enc = dhandler.encode(X_orig_test)
    X_prot_enc = dhandler_protected.encode(X_prot_orig)
    X_prot_test_enc = dhandler_protected.encode(X_prot_orig_test)

    dfX_enc = pd.DataFrame(X_enc)
    dfX_test_enc = pd.DataFrame(X_test_enc)
    dfX_prot = pd.DataFrame(X_prot_enc)
    dfX_prot_test = pd.DataFrame(X_prot_test_enc)
    dfy = pd.Series(y)

    d = X_orig.shape[1]
    d_prot = X_prot_orig.shape[1]
    d_enc = X_enc.shape[1]
    d_ext = X.shape[1]
    d_ext_prot = X_prot.shape[1]

    t_start = time.time()
    vals = cfg.model.split("_")
    bias_type = ""
    group_type = ""
    predictor = vals[0]
    if len(vals) > 1:
        bias_type = vals[1]
        if len(vals) > 2:
            group_type = vals[2]

    if predictor == "dnf":
        # if "msd" in cfg.model:
        #     fairness_type = "MSD"
        #     if "msd0" in cfg.model:
        #         fairness_type = "MSD0"
        # elif "spsf" in cfg.model:
        #     fairness_type = "SPSF"
        # elif "fpsf" in cfg.model:
        #     fairness_type = "FPSF"
        # else:
        #     fairness_type = ""
            # raise ValueError("Incorrectly specified type of fairness.")
        mio_setup = DNFFairClassifier(gamma=cfg.gamma * 4 if bias_type == "msd" else cfg.gamma, fairness_type=bias_type.upper(), subgroup_type=group_type)
        dnf_model = mio_setup.find_dnf(
            X, X_prot if group_type=="rule" else X_prot_enc, y, n_terms=cfg.n_terms, time_limit=cfg.time_limit, verbose=True
        )

        y_hat_train = np.zeros_like(y, dtype=bool)
        for term in dnf_model:
            y_term = np.ones_like(y, dtype=bool)
            for conj in term:
                y_term &= X[:, conj]
            y_hat_train |= y_term
        y_hat_train_prob = y_hat_train.astype(int)

        y_hat_test = np.zeros_like(y_test, dtype=bool)
        for term in dnf_model:
            y_term = np.ones_like(y_test, dtype=bool)
            for conj in term:
                y_term &= X_test[:, conj]
            y_hat_test |= y_term
        y_hat_test_prob = y_hat_test.astype(int)
    elif predictor == "linear":
        # if "msd" in cfg.model:
        #     fairness_type = "MSD"
        #     if "msd0" in cfg.model:
        #         fairness_type = "MSD0"
        # elif "spsf" in cfg.model:
        #     fairness_type = "SPSF"
        # elif "fpsf" in cfg.model:
        #     fairness_type = "FPSF"
        # else:
        #     fairness_type = ""
            # raise ValueError("Incorrectly specified type of fairness.")
        mio_setup = LinearFairClassifier(gamma=(cfg.gamma * 4) if bias_type == "msd" else cfg.gamma, fairness_type=bias_type.upper(), subgroup_type=group_type, sparsity_term=cfg.sparsity_term)
        coefs, threshold = mio_setup.find_classifier(
            X_enc, X_prot if group_type=="rule" else X_prot_enc, y, time_limit=cfg.time_limit, epsilon=1e-3, verbose=True, warmstart=True
        )
        y_hat_train = X_enc @ coefs.reshape((-1, 1)) >= threshold
        y_hat_train = y_hat_train.flatten()
        y_hat_train_prob = y_hat_train.astype(int)

        y_hat_test = X_test_enc @ coefs.reshape((-1, 1)) >= threshold
        y_hat_test = y_hat_test.flatten()
        y_hat_test_prob = y_hat_test.astype(int)
    elif predictor == "gerry":
        # by default, the time limit is only converted to a number of iterations (5 s per iteration);
        # with gerry_wall_clock_limit=true, training is additionally stopped after time_limit seconds
        gerry_time_limit = cfg.time_limit if cfg.get("gerry_wall_clock_limit", False) else None
        if bias_type == "spsf":
            gerryfair_model = Model(printflag=True, gamma=cfg.gamma, fairness_def="SP", time_limit=gerry_time_limit)
        elif bias_type == "fpsf":
            gerryfair_model = Model(printflag=True, gamma=cfg.gamma, fairness_def="FP", time_limit=gerry_time_limit)
        n_iters = cfg.time_limit // 5
        gerryfair_model.set_options(max_iters=n_iters)
        gerryfair_model.train(dfX_enc, dfX_prot, dfy)

        y_hat_train_prob = np.array(gerryfair_model.predict(dfX_enc))
        y_hat_train = y_hat_train_prob >= 0.5

        y_hat_test_prob = np.array(gerryfair_model.predict(dfX_test_enc))
        y_hat_test = y_hat_test_prob >= 0.5
    else:
        raise ValueError(f"Unknown fair classifier {cfg.model}")
    t = time.time()

    print("TRAINED")

    mio = BiasDetect()
    # msd = MSD()
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

    # spsf_mio = SPSF()
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
    mask = ~y_test
    fpsf_test_group_rule = mio.find_subgroup(X_prot_test[mask], y_hat_test[mask], time_limit=cfg.time_limit // 6, bias_type="spsf", group_type="rule")
    ingroup = np.ones_like(y_orig_test, dtype=bool)
    for feat_i in fpsf_test_group_rule:
        ingroup &= X_prot_test[:, feat_i]
    fpsf_mio_val_test = eval_fpsf(y_test, y_hat_test, ingroup)

    # Get the current working directory, which Hydra sets for each run
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Save the output and error logs to a file in the current run directory
    with open(os.path.join(run_dir, "output.txt"), "w") as out_file:
        print(f"Config:\n {cfg}", file=sys.stderr)
        out_file.write(f"Config:\n {cfg}\n")
        out_file.write(f"\nGit hash: {gitcommit}\n\n")
        out_file.write("RESULT\n")
        if "dnf" in cfg.model:
            out_file.write(f"DNF: {dnf_model} \n")
            bin_feats = binarizer.feature_names(include_negations=True)
            out_file.write(f"DNF interp: {[[str(bin_feats[c]) for c in rule] for rule in dnf_model]} \n")
        if "linear" in cfg.model:
            out_file.write(f"Classif Coefs: {list(coefs)} \n")
            out_file.write(f"Classif Threshold: {threshold} \n")
        if "dnf" in cfg.model or "linear" in cfg.model:
            out_file.write(f"Status: {mio_setup.mio_result.solver.status} \n") # probably crashes here in dnfs with linear subgroups?
            out_file.write(
                f"Termination Condition: {mio_setup.mio_result.solver.termination_condition} \n"
            )
            out_file.write(f"Number of cuts: {mio_setup.n_cuts} \n")
            out_file.write(f"Number of callbacks: {mio_setup.n_callbacks} \n")
            out_file.write(
                f"Time in callbacks proportion: {mio_setup.callback_time_proportion} \n"
            )
        out_file.write(f"Train Accuracy: {accuracy(y, y_hat_train)} \n")
        out_file.write(f"Test Accuracy: {accuracy(y_test, y_hat_test)} \n")
        out_file.write(f"Train F1: {f1_score(y, y_hat_train)} \n")
        out_file.write(f"Test F1: {f1_score(y_test, y_hat_test)} \n")
        out_file.write(f"MSD: {msd_val_train} {msd_val_test} \n")
        out_file.write(f"Gerry FPSF: {fpsf_val_train} {fpsf_val_test} \n")
        out_file.write(f"Gerry SPSF: {spsf_val_train} {spsf_val_test} \n")
        out_file.write(f"MIO FPSF: {fpsf_mio_val_train} {fpsf_mio_val_test} \n")
        out_file.write(f"MIO SPSF: {spsf_mio_val_train} {spsf_mio_val_test} \n")

        out_file.write(f"Time: {t-t_start} \n")
        out_file.write(f"True number of training samples: {n_samples} \n")
        out_file.write(f"True number of testing samples: {n_test} \n")
        out_file.write(f"Protected dimension: {d_prot} \n")
        out_file.write(f"Full dimension: {d} \n")
        out_file.write(f"Full encoded dimension: {d_enc} \n")
        out_file.write(f"Protected binarized dimension: {d_ext_prot} \n")
        out_file.write(f"Full binarized dimension: {d_ext} \n")

    print(f"Result saved to {os.path.join(run_dir, 'output.txt')}")
    if "gerry" in cfg.model:
        with open(os.path.join(run_dir, "gerrymodel.pickle"), "wb") as f:
            pickle.dump(gerryfair_model, f)


if __name__ == "__main__":
    # record the code version in output.txt (if run from a git checkout)
    res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    gitcommit = res.stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    if status.stdout.strip() != "":
        print("Warning: uncommitted changes in the repository; results may not match the recorded git hash.", file=sys.stderr)
        gitcommit += " (dirty)"
    run_experiment()
