import os
import pickle
import subprocess
import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from gerryfair.model import Auditor
from scenarios.folktables_scenarios import load_classif_scenario
from bias_detect_mio import BiasDetect
from utils import eval_msd, eval_spsf
import time

gitcommit = ""


@hydra.main(version_base="1.3", config_path="conf", config_name="config_eval")
def run_experiment(cfg: DictConfig):
    binarizer, dhandler, X_orig, y_orig, _, _, binarizer_protected, dhandler_protected, protected_cols = (
        load_classif_scenario(cfg.scenario, cfg.seed, cfg.max_samples)
    )

    X_prot_orig = X_orig[protected_cols]

    n_samples = X_orig.shape[0]
    X = binarizer.encode(X_orig, include_negations=False)
    X_prot = binarizer_protected.encode(X_prot_orig, include_negations=False)
    # X = binarizer.encode(X_orig, include_negations=True)
    # X_prot = binarizer_protected.encode(X_prot_orig, include_negations=True)
    y = binarizer.encode_y(y_orig)

    X_enc = dhandler.encode(X_orig)
    X_prot_enc = dhandler_protected.encode(X_prot_orig)
    # print(X_prot_orig.iloc[0])
    # print(X_prot_enc[0])

    dfX_enc = pd.DataFrame(X_enc)
    dfX_prot = pd.DataFrame(X_prot_enc)
    dfy = pd.Series(y)

    d = X_orig.shape[1]
    d_prot = X_prot_orig.shape[1]
    d_enc = X_enc.shape[1]
    d_prot_enc = X_prot_enc.shape[1]
    d_ext = X.shape[1]
    d_ext_prot = X_prot.shape[1]

    rule = None
    coefs = None
    auditor = None
    t_start = time.perf_counter()
    opt = False
    if "mio" in cfg.method:
        _, bias_type, group_type = cfg.method.split("_")
        detector = BiasDetect()
        if group_type == "rule":
            group_desc, opt = detector.find_subgroup(X_prot, y, time_limit=cfg.time_limit, return_opt_flag=True, bias_type=bias_type, group_type = group_type, n_min=cfg.n_min, warmstart=cfg.warmstart, solver=cfg.solver)
            rule = group_desc
            ingroup = np.ones_like(y, dtype=bool)
            for feat_i in rule:
                ingroup &= X_prot[:, feat_i]
        elif group_type == "linear":
            warm_auditor = Auditor(dfX_prot, np.zeros_like(y), "FP")
            ingroup, warm_oracle = warm_auditor.audit(np.array(y_orig), with_group_def=True)
            coeffs = warm_oracle.b1.coef_
            thresh = -warm_oracle.b1.intercept_
            m = max(np.abs(coeffs).max(), np.abs(thresh) / d_prot_enc)
            vals = X_prot_enc @ (coeffs/m) - (thresh/m)
            if np.min(vals[vals > 0]) - np.max(vals[vals <= 0]) > 1e-6 or cfg.skip_gerry_init:
                warm_model = {
                    "coeffs": list(coeffs / m),
                    "thresh": float(thresh / m),
                }
                group_desc, opt = detector.find_subgroup(X_prot_enc, y, time_limit=cfg.time_limit, return_opt_flag=True, bias_type=bias_type, group_type = group_type, verbose=True, warmstart_model=warm_model, n_min=cfg.n_min, warmstart=cfg.warmstart, solver=cfg.solver)
            else:
                group_desc, opt = detector.find_subgroup(X_prot_enc, y, time_limit=cfg.time_limit, return_opt_flag=True, bias_type=bias_type, group_type = group_type, verbose=True, n_min=cfg.n_min, warmstart=cfg.warmstart, solver=cfg.solver)
            coefs, thresh = group_desc[:-1], group_desc[-1]
            ingroup = X_prot_enc @ coefs >= thresh
    elif cfg.method == "gerry_spsf":
        auditor = Auditor(dfX_prot, np.zeros_like(y), "FP")
        ingroup, oracle = auditor.audit(np.array(y_orig), with_group_def=True)
        ingroup = np.array(ingroup, dtype=bool)
    else:
        raise ValueError(f"Unknown method {cfg.method}")
    t = time.perf_counter()


    # Get the current working directory, which Hydra sets for each run
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Save the output and error logs to a file in the current run directory
    with open(os.path.join(run_dir, "output.txt"), "w") as out_file:
        print(f"Config:\n {cfg}", file=sys.stderr)
        out_file.write(f"Config:\n {cfg}\n")
        out_file.write(f"\nGit hash: {gitcommit}\n\n")
        out_file.write("RESULT\n")
        if rule is not None:
            out_file.write(f"Rule group: {rule} \n")
            # bin_feats = binarizer_protected.feature_names(include_negations=True)
            bin_feats = binarizer_protected.feature_names(include_negations=False)
            out_file.write(f"Rule group interp: {[str(bin_feats[c]) for c in rule]} \n")
        if coefs is not None:
            out_file.write(f"Linear group: x T coefs >= thresh \n")
            # bin_feats = binarizer_protected.feature_names(include_negations=True)
            out_file.write(f"Linear group coeficients: {list(coefs)} \n")
            out_file.write(f"Linear group threshold: {thresh} \n")
        if auditor is not None:
            out_file.write(f"Gerry oracle b0 coef: {list(oracle.b0.coef_)} \n")
            out_file.write(f"Gerry oracle b0 intercept: {oracle.b0.intercept_} \n")
            out_file.write(f"Gerry oracle b1 coef: {list(oracle.b1.coef_)} \n")
            out_file.write(f"Gerry oracle b1 intercept: {oracle.b1.intercept_} \n")
            out_file.write(f"Columns: {dhandler_protected.get_feature_names(encoded=True)} \n")
        out_file.write(f"SPSF: {eval_spsf(y, ingroup)} \n")
        out_file.write(f"MSD: {eval_msd(y, ingroup)} \n")
        out_file.write(f"Time: {t-t_start} \n")
        out_file.write(f"Optimal: {opt} \n")
        out_file.write(f"True number of training samples: {n_samples} \n")
        out_file.write(f"Protected dimension: {d_prot} \n")
        out_file.write(f"Full dimension: {d} \n")
        out_file.write(f"Full encoded dimension: {d_enc} \n")
        out_file.write(f"Protected binarized dimension: {d_ext_prot} \n")
        out_file.write(f"Full binarized dimension: {d_ext} \n")


if __name__ == "__main__":
    # record the code version in output.txt (if run from a git checkout)
    res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    gitcommit = res.stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    if status.stdout.strip() != "":
        print("Warning: uncommitted changes in the repository; results may not match the recorded git hash.", file=sys.stderr)
        gitcommit += " (dirty)"
    run_experiment()
