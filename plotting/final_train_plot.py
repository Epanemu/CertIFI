import os
import re
import sys
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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

base_dir_prefix = (
    # "batch_from_cluster/train/experiment_train"
    # "batch_out/train/experiment_train"
    # "batch_from_cluster/train_vanoce"
    # "batch_from_cluster/train_silvestr_fixed"
    # "batch_from_cluster/train_final"
    # "batch_from_cluster/train_final3"
    # "batch_from_cluster/train_final_linear_sparse01"
    # "batch_from_cluster/train_final_linear_nosparse"

    "batch_from_cluster/neurips_train_long"
)

method_order = [
    "linear",               # base model
    "linear_fpsf_rule",
    "linear_msd0_rule",
    "linear_spsf_rule",
    "linear_msd_rule",
    "linear_fpsf_linear",
    "linear_msd0_linear",
    "linear_spsf_linear",
    "linear_msd_linear",
    "gerry_fpsf"            # existing work
]

method_labels = {
    "linear":             "No fairness",
    "linear_fpsf_linear": "MIO - FPSF directly",
    "linear_fpsf_rule":   "MIO - FPSF directly",
    "linear_msd0_linear": "MIO - FPSF by SD",
    "linear_msd0_rule":   "MIO - FPSF by SD",
    "gerry_fpsf":         "GerryFair (FPSF)",
    "linear_msd_rule": "MIO - MSD directly",
    "linear_spsf_rule": "MIO - SPSF directly",
    "linear_msd_linear": "MIO - MSD directly",
    "linear_spsf_linear": "MIO - SPSF directly",
}

method_colors = {
    "linear": "#888888",
    "linear_fpsf_linear": "#F0C220",
    "linear_fpsf_rule":   "#F0C220",
    "linear_msd0_linear": "#0173B2",
    "linear_msd0_rule":   "#0173B2",
    "gerry_fpsf":         "#DE8F05",
    "linear_msd_rule": "red",
    "linear_spsf_rule": "green",
    "linear_msd_linear": "red",
    "linear_spsf_linear": "green",
}

method_hatches = {
    "linear": "",
    "linear_fpsf_linear": "",
    "linear_fpsf_rule": "//",
    "linear_msd0_linear": "",
    "linear_msd0_rule": "//",
    "gerry_fpsf": "",
    "linear_msd_rule": "//",
    "linear_spsf_rule": "//",
    "linear_msd_linear": "",
    "linear_spsf_linear": "",
}

method_markers = {
    "linear": "o",
    "linear_fpsf_linear": "o",
    "linear_fpsf_rule": "^",
    "linear_msd0_linear": "s",
    "linear_msd0_rule": "v",
    "gerry_fpsf": "D",
}

cut_vals = []

def extract_data():
    extracted_data = []
    base_dir = base_dir_prefix

    for method in method_order:
        for scenario in scenarios:
            # for i in range(5):
            for i in range(3): # just 3 seeds
                # if method == "linear":
                    # folder_path = os.path.join(base_dir, f"{method}-{scenario}-0", str(i))
                # if "_msd_" in method:
                #     folder_path = os.path.join(base_dir, f"{method}-{scenario}-0.1", str(i))
                # else:
                # folder_path = os.path.join(base_dir, f"{method}-{scenario}-0.01", str(i))
                folder_path = os.path.join(base_dir, f"{method}-{scenario}-5", str(i))
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
                            extracted_data.append(("Accuracy", scenario, method, acc_val))
                        f1 = re.search(r"Test F1: ([\d\.]+)", line)
                        if f1:
                            f1_val = float(f1.group(1))
                            extracted_data.append(("F$_1$ score", scenario, method, f1_val))
                        spsf = re.search(r"MIO FPSF: ([\d\.]+)", line)
                        if spsf:
                            spsf_val = float(spsf.group(1))
                            extracted_data.append(("FPSF (train)", scenario, method, spsf_val))
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
                            extracted_data.append(("FPSF (test)", scenario, method, fpsf_val))
                        cuts = re.search(r"Number of cuts: ([\d\.]+)", line)
                        if cuts:
                            cut_val = float(cuts.group(1))
                            # print("cuts", method, scenario, cut_val)
                            cut_vals.append((method, scenario, cut_val))
                            extracted_data.append(("Number of cuts", scenario, method, cut_val))
                        time = re.search(r"Time: ([\d\.]+)", line)
                        if time:
                            time_val = float(time.group(1))
                            extracted_data.append(("Time [s]", scenario, method, time_val))

    data_dict = {}

    for (
        valname,
        scenario,
        method,
        val,
    ) in extracted_data:
        if method not in data_dict:
            data_dict[method] = {}
        if valname not in data_dict[method]:
            data_dict[method][valname] = defaultdict(list)
        data_dict[method][valname][scenario].append(val)

    return data_dict


all_data = extract_data()

# r, c = 1, 5
# r, c = 6, 5
r, c = 6, 5
fig, axs = plt.subplots(r, c, figsize=(12, 12))
# fig, axs = plt.subplots(r, c, figsize=(9, 6))
# fig, axs = plt.subplots(r, c, figsize=(12, 5))

# ax = axs[0,0]

for j, scenario in enumerate(scenarios):
# for k, val in enumerate(["Accuracy", "F1", "MSD", "SPSF", "FPSF", "Time"]):
    acc_data = []
    train_data = []
    test_data = []
    cut_data = []
    time_data = []
    f1_data = []

    for i, method in enumerate(method_order):
        if method not in all_data:
            continue
        data_dict = all_data[method]
        # for k, val in enumerate(all_data[methods[0]].keys()):
        ax_acc = axs[0, j % c]
        ax_f1 = axs[1, j % c]
        ax_train = axs[2, j % c]
        ax_test = axs[3, j % c]
        ax_cuts = axs[4, j % c]
        ax_time = axs[5, j % c]
        # ax = axs[j % c]

        accs = np.array(data_dict["Accuracy"][scenario])
        fair_test = np.array(data_dict["FPSF (test)"][scenario])
        fair_train = np.array(data_dict["FPSF (train)"][scenario])
        f1 = np.array(data_dict["F$_1$ score"][scenario])
        time = np.array(data_dict["Time [s]"][scenario])
        cut = np.array(data_dict["Number of cuts"][scenario]) if "Number of cuts" in data_dict else np.zeros_like(time)
        if sys.argv[1] == "plot":

            fair_threshold = 0.005 if "Mobility" in scenario else 0.01
            # if i == 0:
            #     ax_scatter.axvline(
            #         x=fair_threshold,
            #         linestyle="--",
            #         linewidth=1.2,
            #         color="black",
            #         label="Fairness target ($\\gamma$)"
            #     )


            # ax_scatter.scatter(
            #         fair_test,
            #         accs,
            #         c=method_colors[method],
            #         marker=method_markers[method],
            #         s=40,
            #         edgecolors="black",
            #         alpha=0.85,
            #         label=method_labels[method],
            #     )


            # ax_scatter.set_xlabel("FPSF violation (test)")
            # if j == 0:
            #     ax_scatter.set_ylabel("Test accuracy")
            # ax_scatter.set_title(scenario_titles[scenario])

            # ax_scatter.spines["top"].set_visible(False)
            # ax_scatter.spines["right"].set_visible(False)
            # ax_scatter.grid(axis="y", alpha=0.3)

            # if j == 4:
            #     ax_scatter.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

            # # =========================
            # # ROW 2 — BOX PLOTS OF FPSF (TRAIN)
            # # =========================

            # # box_data = [fair_train[method] for method in method_order]

            # bp = ax_box.boxplot(
            #     [fair_train],
            #     positions=[i],
            #     patch_artist=True,
            #     showmeans=True,
            #     meanprops=dict(marker="o", markerfacecolor="black")
            # )

            # # Apply colors + hatches consistently
            # for i, box in enumerate(bp["boxes"]):
            #     method = method_order[i]
            #     box.set_facecolor(method_colors[method])
            #     box.set_edgecolor("black")
            #     box.set_hatch(method_hatches[method])

            # # make whiskers/medians black
            # for key in ["whiskers", "caps", "medians"]:
            #     for line in bp[key]:
            #         line.set_color("black")

            # ax_box.axhline(
            #     y=fair_threshold,
            #     linestyle="--",
            #     linewidth=1.2,
            #     color="black"
            # )

            # ax_box.set_xticks(range(len(method_order)))
            # ax_box.set_xticklabels(
            #     [method_labels[method] for method in method_order],
            #     rotation=30,
            #     ha="right"
            # )

            # ax_box.set_ylabel("FPSF violation (train)")
            # ax_box.set_title("Fairness Satisfaction on Training Data")

            # ax_box.spines["top"].set_visible(False)
            # ax_box.spines["right"].set_visible(False)
            # ax_box.grid(axis="y", alpha=0.3)
            acc_data.append(accs)
            train_data.append(fair_train)
            test_data.append(fair_test)
            cut_data.append(cut)
            time_data.append(time)
            f1_data.append(f1)

    def styled_boxplot(ax, data, title="", show_fair_line=False, ylabel=None):
        bp = ax.boxplot(
            data,
            patch_artist=True,
            showmeans=False,
            widths=0.6
        )

        # Style boxes
        for i, box in enumerate(bp["boxes"]):
            m = method_order[i]
            box.set_facecolor(method_colors[m])
            # box.set_edgecolor("black")
            box.set_linewidth(0)
            box.set_hatch(method_hatches[m])

        # Make whiskers/medians clean and bold
        for key in ["whiskers", "caps", "medians"]:
            for line in bp[key]:
                line.set_color("black")
                line.set_linewidth(1.2)

        if title != "":
            ax.set_title(title, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.8)
            ax.spines[spine].set_color("#444444")
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks([])

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if show_fair_line:
            ax.axhline(
                y=fair_threshold,
                linestyle="--",
                linewidth=1.2,
                color="black",
                label="Fairness target"
            )

        return bp

    # --------- Make the three panels ----------
    bp1 = styled_boxplot(
        ax_acc, acc_data,
        scenario_titles[scenario],
        ylabel="Test Accuracy" if j == 0 else None
    )

    bp2 = styled_boxplot(
        ax_train, train_data,
        ylabel="FPSF violation (train)" if j == 0 else None,
        show_fair_line=True
    )

    bp3 = styled_boxplot(
        ax_test, test_data,
        ylabel="FPSF violation (test)" if j == 0 else None,
        show_fair_line=True
    )

    bp4 = styled_boxplot(
        ax_cuts, cut_data,
        ylabel="Number of cuts" if j == 0 else None,
    )

    bp5 = styled_boxplot(
        ax_time, time_data,
        ylabel="Time [s]" if j == 0 else None,
    )

    bp6 = styled_boxplot(
        ax_f1, f1_data,
        ylabel="F1 score (test)" if j == 0 else None,
    )

    _, maxy = ax_time.get_ylim()
    ax_time.set_ylim(0, maxy)

    # Shared x-ticks
    # axes[1].set_xticks(range(1, len(method_order) + 1))
    # axes[1].set_xticklabels(
    #     [method_labels[m] for m in method_order],
    #     rotation=30,
    #     ha="right"
    # )

# --------- Single shared legend ----------
from matplotlib.patches import Patch

handles = [
    Patch(color='none', label=''),
    Patch(facecolor='white', hatch="//",  label=r'$\bf{Conjunction\ Subgroups}$'),#\ (\mathcal{S}_A)}$'),
    Patch(color='none', label=''),
    Patch(color='none', label=r'$\bf{Linear\ Subgroups}$'),#\ (\mathcal{S}_L)}$'),
    Patch(color='none', label=''),
    Patch(color='none', label=''),
]
legend_handles = []
for i, m in enumerate(method_order):
    if i % 2 == 1 or i == 0:
        legend_handles.append(handles[(i+1)//2])
    legend_handles.append(Patch(
        facecolor=method_colors[m],
        # edgecolor="black",
        hatch=method_hatches[m],
        label=method_labels[m]
    ))
    if i == 0:
        # legend_handles.append(Patch(color='none', label=''))
        # legend_handles.append(Patch(color='none', label=''))
        # legend_handles.append(Patch(color='none', label=''))
        legend_handles.append(Patch(color='none', label=''))
    if i == 4:
        legend_handles.append(Patch(color='none', label=''))
        legend_handles.append(Patch(color='none', label=''))
        legend_handles.append(Patch(color='none', label=''))



fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=7,
    # frameon=False,
    bbox_to_anchor=(0.5, 0),
    handletextpad=0.5, columnspacing=1.5, fontsize=10
)

# fig.legend(handles, labels, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1))
# handles, labels = axs[0,0].get_legend_handles_labels()
# handles, labels = ax.get_legend_handles_labels()

plt.tight_layout(rect=[0, 0.06, 1, 1])
# plt.show()

# handles = [
#     mpatches.Patch(color=method_colors[m], label=method_names[m])
#     for m in method_order
# ]
# labels = [method_names[m] for m in methods]
# fig.legend(
#     handles, labels, loc="lower center", ncol=len(methods), bbox_to_anchor=(0.5, 0)
# )

# # plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.tight_layout(rect=[0, 0.05, 1, 1])
output_path = (
    "plots/" + str(date.today()) + f"_train.pdf"
)
plt.savefig(output_path)

plt.show()

import pandas as pd
df = pd.DataFrame(cut_vals, columns=['Method', 'Dataset', 'Value'])

print()
print(f"MAX CUTS: {df['Value'].max()}")
print()

# Helper to format mean +- std
def format_stat(x):
    return f"{x.mean():.2f} $\pm$ {x.std():.2f}"

# Calculate main pivot table
pivot = df.pivot_table(
    index='Method',
    columns='Dataset',
    values='Value',
    aggfunc=format_stat
)

# Calculate marginals (Totals)
method_totals = df.groupby('Method')['Value'].apply(format_stat)
dataset_totals = df.groupby('Dataset')['Value'].apply(format_stat)
grand_total = format_stat(df['Value'])

# Add Marginal Column and Row
pivot['Total (Method)'] = method_totals
pivot.loc['Total (Dataset)'] = dataset_totals
pivot.loc['Total (Dataset)', 'Total (Method)'] = grand_total

# Clean up names for LaTeX
pivot.index = [i.replace('_', r'\_') for i in pivot.index]
pivot.columns = [c.replace('_', r'\_') for c in pivot.columns]

print(pivot.to_latex(escape=False, column_format='l' + 'c'*len(pivot.columns)))

print()
print()
print()

df = pd.DataFrame(cut_vals, columns=['Method', 'Dataset', 'Value'])
df["Method"] = df["Method"].apply(lambda x: "rule" if "_rule" in x else "linear")

# Helper to format mean +- std
def format_stat(x):
    return f"{x.mean():.2f} $\pm$ {x.std():.2f}"

# Calculate main pivot table
pivot = df.pivot_table(
    index='Method',
    columns='Dataset',
    values='Value',
    aggfunc=format_stat
)

# Calculate marginals (Totals)
method_totals = df.groupby('Method')['Value'].apply(format_stat)
dataset_totals = df.groupby('Dataset')['Value'].apply(format_stat)
grand_total = format_stat(df['Value'])

# Add Marginal Column and Row
pivot['Total (Method)'] = method_totals
pivot.loc['Total (Dataset)'] = dataset_totals
pivot.loc['Total (Dataset)', 'Total (Method)'] = grand_total

# Clean up names for LaTeX
pivot.index = [i.replace('_', r'\_') for i in pivot.index]
pivot.columns = [c.replace('_', r'\_') for c in pivot.columns]

print(pivot.T.to_latex(escape=False, column_format='l' + 'c'*len(pivot.columns)))