import os
import re
import sys
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

scenarios = [
    "ACSIncome",
    "ACSPublicCoverage",
    "ACSMobility",
    "ACSEmployment",
    "ACSTravelTime",
]
scenario_titles = {
    "ACSIncome": "ACS Income",
    "ACSPublicCoverage": "ACS Public Coverage",
    "ACSMobility": "ACS Mobility",
    "ACSEmployment": "ACS Employment",
    "ACSTravelTime": "ACS Travel Time",
}

base_dir_dense = "batch_from_cluster/train_final_linear_nosparse"
base_dir_sparse = "batch_from_cluster/train_final_linear_sparse01"

method_sets = [
    ["linear_fpsf_rule", "linear_msd0_rule"],
    ["linear_fpsf_linear", "linear_msd0_linear"],
]

method_labels = {
    "linear":             "No fairness",
    "linear_fpsf_linear": "MIO - FPSF directly",
    "linear_fpsf_rule":   "MIO - FPSF directly",
    "linear_msd0_linear": "MIO - FPSF by SD",
    "linear_msd0_rule":   "MIO - FPSF by SD",
    "gerry_fpsf":         "GerryFair (FPSF)"
}

method_colors = {
    "linear": "#888888",
    "linear_fpsf_linear": "#F0C220",
    "linear_fpsf_rule":   "#F0C220",
    "linear_msd0_linear": "#0173B2",
    "linear_msd0_rule":   "#0173B2",
    "gerry_fpsf":         "#DE8F05",
}

method_hatches = {
    "linear": "",
    "linear_fpsf_linear": "",
    "linear_fpsf_rule": "//",
    "linear_msd0_linear": "",
    "linear_msd0_rule": "//",
    "gerry_fpsf": "",
}

method_markers = {
    "linear": "o",
    "linear_fpsf_linear": "o",
    "linear_fpsf_rule": "^",
    "linear_msd0_linear": "s",
    "linear_msd0_rule": "v",
    "gerry_fpsf": "D",
}

set_name = ["Conjunction Subgroups", "Linear Model Subgroups"]

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
# fig, ax = plt.subplots(2, 1, figsize=(5, 7), sharey=True)

for seti, method_order in enumerate(method_sets):
    def extract_data(base_dir):
        extracted_data = []

        for method in method_order:
            for scenario in scenarios:
                for i in range(5):
                    # if method == "linear":
                        # folder_path = os.path.join(base_dir, f"{method}-{scenario}-0", str(i))
                    # if "_msd_" in method:
                    #     folder_path = os.path.join(base_dir, f"{method}-{scenario}-0.1", str(i))
                    # else:
                    folder_path = os.path.join(base_dir, f"{method}-{scenario}-0.01", str(i))
                    output_file = os.path.join(folder_path, "output.txt")
                    if not os.path.isfile(output_file):
                        print(f"passing {i} {output_file}")
                        continue

                    with open(output_file, "r", errors="ignore") as file:
                        lines = file.readlines()
                        for line in lines:
                            acc = re.search(r"Test Accuracy: ([\d\.]+)", line)
                            if acc:
                                acc_val = float(acc.group(1))
                                extracted_data.append(("Accuracy", scenario, method, acc_val, i))
                            f1 = re.search(r"Test F1: ([\d\.]+)", line)
                            if f1:
                                f1_val = float(f1.group(1))
                                # extracted_data.append(("F$_1$ score", scenario, method, f1_val, i))
                            spsf = re.search(r"MIO FPSF: ([\d\.]+)", line)
                            if spsf:
                                spsf_val = float(spsf.group(1))
                                extracted_data.append(("FPSF (train)", scenario, method, spsf_val, i))
                            # spsf = re.search(r"MIO SPSF: [\d\.]+ ([\d\.]+)", line)
                            # if spsf:
                            #     spsf_val = float(spsf.group(1))
                            #     extracted_data.append(("SPSF", scenario, method, spsf_val))
                            # msd = re.search(r"MSD: [\d\.]+ ([\d\.]+)", line)
                            # if msd:
                            #     msd_val = float(msd.group(1))
                            #     extracted_data.append(("MSD", scenario, method, msd_val))
                            fpsf = re.search(r"MIO FPSF: [\d\.]+ ([\d\.]+)", line)
                            if fpsf:
                                fpsf_val = float(fpsf.group(1))
                                extracted_data.append(("FPSF (test)", scenario, method, fpsf_val, i))
                            time = re.search(r"Number of cuts: ([\d\.]+)", line)
                            if time:
                                time_val = float(time.group(1))
                                # print("cuts", method, scenario, time_val)
                                extracted_data.append(("# Cuts", scenario, method, time_val, i))
                                # cut_vals.append((method, scenario, time_val))
                                # extracted_data.append(("Cuts", scenario, method, time_val))
                            match = re.search(r"Classif Coefs: \[(.*?)\]", line)
                            if match:
                                coef_string = match.group(1)

                                # 2. Extract all numbers as floats
                                # This regex handles negatives, decimals, and integers
                                numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", coef_string)
                                coefs = [float(n) for n in numbers]

                                # 3. Calculate proportion of zeros
                                num_zeros = sum(1 for x in coefs if x == 0.0)
                                total = len(coefs)
                                sparsity.append(num_zeros / total if total > 0 else 0)
                                # extracted_data.append(("Sparsity", scenario, method, proportion, i))
                            # time = re.search(r"Time: ([\d\.]+)", line)
                            # if time:
                            #     time_val = float(time.group(1))
                            #     extracted_data.append(("Time", scenario, method, time_val))

        data_dict = {}

        # for (
        #     valname,
        #     scenario,
        #     method,
        #     val,
        # ) in extracted_data:
        #     if method not in data_dict:
        #         data_dict[method] = {}
        #     if valname not in data_dict[method]:
        #         data_dict[method][valname] = defaultdict(list)
        #     data_dict[method][valname][scenario].append(val)

        return pd.DataFrame(extracted_data, columns=["quantity_name", "dataset", "method", "value", "seed"])

    sparsity = []
    all_data_dense = extract_data(base_dir_dense)
    print(f"Mean sparsity - dense ({set_name[seti]}):", np.mean(sparsity), "+-", np.std(sparsity))
    sparsity = []
    all_data_sparse = extract_data(base_dir_sparse)
    print(f"Mean sparsity - sparse ({set_name[seti]}):", np.mean(sparsity), "+-", np.std(sparsity))

    all_data_dense['Config'] = 'Dense'
    all_data_sparse['Config'] = 'Sparse'
    df_full = pd.concat([all_data_dense, all_data_sparse], ignore_index=True)

    # Pivot for Scatter Plot (X=Violation, Y=Accuracy)
    # scatter_df = df_full.pivot_table(
    #     index=['dataset', 'seed', "method", 'Config'],
    #     columns='quantity_name',
    #     values='value'
    # ).reset_index()

    # Align seeds to calculate differences (Sparse - Dense)
    diff_df = pd.merge(
        all_data_dense, all_data_sparse,
        on=['quantity_name', 'dataset', "method", 'seed'],
        suffixes=('_dense', '_sparse')
    )
    diff_df['diff'] = (diff_df['value_sparse'] - diff_df['value_dense'])/diff_df['value_sparse']

    # # 2. Statistical Test (Paired T-test)
    # stats_results = []
    # for (ds, q), group in diff_df.groupby(['dataset', 'quantity_name']):
    #     # ttest_rel compares the same seeds across two conditions
    #     t_stat, p_val = stats.ttest_rel(group['value_sparse'], group['value_dense'])
    #     stats_results.append({
    #         'Dataset': ds, 'Metric': q,
    #         'Mean Diff': group['diff'].mean(), 'p-value': p_val
    #     })
    # stats_table = pd.DataFrame(stats_results)
    # print(stats_table)


    # Plot A: Scatter Trade-off
    # sns.scatterplot(
    #     data=scatter_df, x='FPSF (test)', y='Accuracy',
    #     hue='Config', style='dataset', s=100, ax=ax[0]
    # )
    # ax[0].set_title("Fairness-Accuracy Trade-off per Seed")
    # ax[0].grid(True, linestyle='--', alpha=0.5)

    # Plot B: Difference Impact (Sparse - Dense)
    sns.barplot(
        data=diff_df, x='dataset', y='diff', hue='quantity_name',
        errorbar='sd', capsize=0, ax=ax[seti], zorder=2, linewidth=0.5,
        palette={'Accuracy': '#2ca25f', 'FPSF (train)': '#fc9272', 'FPSF (test)': '#de2d26', '# Cuts': '#e574bc'},
    )
    ax[seti].axhline(0, color='black', lw=1.5, ls='--')
    ax[seti].set_title(set_name[seti])
    # ax[seti].set_ylabel("Relative change (1 - Dense/Sparse)")
    ax[seti].set_ylabel("")
    # if seti == 0:
        # ax[seti].set_xticks([])
    # if seti == 1:
        # ax[seti].set_yticks([])
    # else:
    # ax[seti].tick_params(axis='y', labelleft=True, left=True)
    # plt.xticks(rotation=30, ha="right")
    # plt.xticks(rotation=30)
    ax[seti].tick_params(axis='x', rotation=20)

    ax[seti].set_xlabel("")
    ax[seti].get_legend().remove()
    ax[seti].grid(True, axis='y', linestyle='-', alpha=0.5, zorder=0)
    # ax[seti].spines["top"].set_visible(False)
    # ax[seti].spines["right"].set_visible(False)
    for spine in ["top", "left", "bottom", "right"]:
        ax[seti].spines[spine].set_linewidth(0.8)
        ax[seti].spines[spine].set_color("#444444")

# fig.text(0.01, 0.6, 'Relative change of value (1 - Dense/Sparse)', va='center', rotation='vertical', fontsize=12)
fig.text(0.01, 0.54, 'Relative change of value (1 - Dense/Sparse)', va='center', rotation='vertical', fontsize=12)
handles, labels = ax[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0), columnspacing=1.5)
fig.legend(handles, labels, loc='center right', ncol=1, bbox_to_anchor=(1, 0.6))

plt.tight_layout(rect=[0.02,0.05,0.88,1])
# plt.savefig('plots/sparsity_analysis.pdf')
plt.savefig('plots/sparsity_analysis_wide.pdf')
plt.show()