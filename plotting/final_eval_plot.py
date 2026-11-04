import os
import re
import sys
from collections import defaultdict
from datetime import date
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "DejaVu Sans"
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["DejaVu Serif"]

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

methods = [
    # "msd",
    # "spsf_mio",
    # "spsf",
    "mio_msd_rule",
    "mio_spsf_rule",
    "mio_msd_linear",
    "mio_spsf_linear",
    "gerry_spsf",
]

base_dir_prefix = (
    # "batch_from_cluster/eval/experiment_eval"
    # "batch_from_cluster/eval2/experiment_eval"
    # "batch_from_cluster/eval3/experiment_eval"
    # "batch_from_cluster/eval3/eval/experiment_eval"
    # "batch_from_cluster/eval4/experiment_eval"
    # "batch_from_cluster/eval5"
    # "batch_from_cluster/eval_gerrywarm"

    # "batch_from_cluster/eval_vanoce"
    # "batch_from_cluster/eval_5min"
    # "batch_from_cluster/eval_10min"
    # "batch_from_cluster/eval_small_opt"

    "batch_from_cluster/neurips_eval"
)


colors = ["#4C72B0", "#DD8452", "#55A868"]

method_colors = {
    # "msd": "red",
    # "spsf_mio": "green",
    # "spsf": "blue",
    # "mio_msd_rule": "#91bfdb",
    # "mio_spsf_rule": "#ffffbf",
    # "mio_msd_linear": "#91bfdb",
    # "mio_spsf_linear": "#ffffbf",
    # "gerry_spsf": "#fc8d59",
    "mio_msd_rule": "#0173B2",
    "mio_spsf_rule": "#F0C220",
    "mio_msd_linear": "#0173B2",
    "mio_spsf_linear": "#F0C220",
    "gerry_spsf": "#DE8F05",
}

method_fill = {
    # "msd": "red",
    # "spsf_mio": "green",
    # "spsf": "blue",
    "mio_msd_rule": "red",
    "mio_spsf_rule": "lightblue",
    "mio_msd_linear": "red",
    "mio_spsf_linear": "lightblue",
    "gerry_spsf": "darkblue",
}

method_names = {
    # "msd": "Subgroup as a rule (MSD via MIO)",
    # "spsf_mio": "Subgroup as a rule (SPSF via MIO)",
    # "spsf": "Subgroup as a linear model (SPSF via GerryFair)",
    # "msd": "MSD rule (via MIO)",
    # "spsf_mio": "SPSF rule (via MIO)",
    # "spsf": "SPSF linear model (via GerryFair)",
    "mio_msd_rule": "MIO - SD objective (MSD)",
    "mio_spsf_rule": "MIO - SPSF objective",
    "mio_msd_linear": "MIO - SD objective",
    "mio_spsf_linear": "MIO - SPSF objective",
    "gerry_spsf": "GerryFair - SPSF",
}


def extract_data():
    extracted_data = []
    base_dir = base_dir_prefix

    for method in methods:
        for scenario in scenarios:
            for i in range(1): # all data
            # for i in range(5):
                folder_path = os.path.join(base_dir, f"{method}-{scenario}", str(i))
                output_file = os.path.join(folder_path, "output.txt")
                if not os.path.isfile(output_file):
                    print(f"passing {i} {output_file}")
                    continue

                with open(output_file, "r", errors="ignore") as file:
                    lines = file.readlines()
                    for line in lines:
                        # spsf = re.search(r"SPSF: ([\d\.]+)", line)
                        spsf = re.search(r"SPSF: (.+)", line)
                        if spsf:
                            spsf_val = float(spsf.group(1))
                            extracted_data.append(("Discovered SPSF violation", scenario, method, spsf_val))
                        # msd = re.search(r"MSD: ([\d\.]+)", line)
                        msd = re.search(r"MSD: (.+)", line)
                        if msd:
                            msd_val = float(msd.group(1))
                            # extracted_data.append(("MSD violation", scenario, method, msd_val))
                        # time = re.search(r"Time: ([\d\.]+)", line)
                        time = re.search(r"Time: (.+)", line)
                        if time:
                            time_val = float(time.group(1))
                            # extracted_data.append(("Time [s]", scenario, method, time_val))

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

r, c = 1, 5
# r, c = 3, 5
# fig, axs = plt.subplots(r, c, figsize=(15, 6))
# fig, axs = plt.subplots(r, c, figsize=(12, 5))
fig, axs = plt.subplots(r, c, figsize=(12, 2.8))

# ax = axs[0,0]
x_coords = np.empty((5,))
y_coords = np.zeros((5,))

for k, val in enumerate(all_data[methods[0]].keys()):
    for i, method in enumerate(methods):
        if method not in all_data:
            continue
        data_dict = all_data[method]
        for j, scenario in enumerate(scenarios):
            if scenario not in data_dict[val]:
                continue
            # ax = axs[k, j % c]
            ax = axs[j % c]

            vals = np.array(data_dict[val][scenario])
            print(len(vals))
            sorted_mean = [
                np.mean(vals)
            ]
            sorted_std = [
                np.std(vals)
            ]

            if sys.argv[1] == "plot":
                # if i == 0:
                #     ax.axvspan(-0.5, 1.5, facecolor='#f2f2f2', alpha=0.5, zorder=0)

                x_pos = float(i)
                # 1. Gap between first 2 (0,1) and remaining 3 (2,3,4)
                # if i >= 2:
                #     x_pos += 0.4  # Significant gap
                # # 2. Smaller gap between the 4th (3) and fifth (4)
                # if i == 4:
                #     x_pos += 0.2 # Smaller additional gap

                bar = ax.bar(
                    x_pos,
                    sorted_mean,
                    yerr=sorted_std,
                    # marker="x",
                    # linestyle=method_lines[method],
                    width = 0.6,
                    edgecolor="black",
                    linewidth=0,
                    color=method_colors[method],
                    label=f"{method_names[method]}",
                    zorder=2,
                )
                if i < 2:
                    bar[0].set_hatch("//")
                # if i % 2 == 1:
                #     bar[0].set_hatch("x")
                # elif i == 4:
                #     bar[0].set_hatch("/")
                # else:
                #     bar[0].set_hatch("\\")

                if j % c == 0:
                    ax.set_ylabel(val)
                # if j // c == r-1:
                #     ax.set_xlabel("Number of samples")
                # ax.set_xscale("log")
                if k == 0:
                    ax.set_title(scenario_titles[scenario])
                ax.grid(axis='y', alpha=0.3, linewidth=0.5, color='gray', linestyle='-')
                ax.set_xticks([])
                legend = ax.get_legend_handles_labels()
                handles, labels = [], []
                for h, l in zip(*legend):
                    if l != "":
                        handles.append(h)
                        labels.append(l)

                x_coords[i] = x_pos
                y_coords[j] = max(y_coords[j], sorted_mean[0])

                if i == 4:
                    _, xmax = ax.get_xlim()
                    # ax.set_xlim(-0.7, xmax)
                    # ax.axvspan(-0.7, 1.7, color="#747474", zorder=0)
                    # ax.axvspan(-0.7, 1.7, color="#ACACAC", zorder=0)

                    y = y_coords[j] * 1.07

                    # g1_center = (x_coords[0] + x_coords[1]) / 2
                    # ax.text(g1_center, y, "Conjunction\nsubgroups",
                    #         ha='center', va='center', fontsize=10, color="#ffffff")
                    # # ax.plot([x_coords[0]-0.3, x_coords[1]+0.3], [y/1.01, y/1.01], color='#be123c', lw=1.5)

                    # # Group 2 (Indices 2, 3, 4): Linear Models
                    # g2_center = (x_coords[2] + x_coords[4]) / 2
                    # ax.text(g2_center, y, "Linear subgroups",
                    #         ha='center', va='bottom', fontsize=10, color="#303030")
                    # # ax.plot([x_coords[2]-0.3, x_coords[4]+0.3], [y/1.01, y/1.01], color='#475569', lw=1.5)
                    # ymin, _ = ax.get_ylim()
                    # ax.set_ylim(ymin, y*1.12)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                for spine in ["left", "bottom"]:
                    ax.spines[spine].set_linewidth(0.8)
                    ax.spines[spine].set_color("#444444")
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            else:
                print("specify what type of an output you want - plot, table")
                exit()

legend_elements = [
    # Section 1 Header
    Patch(facecolor='white', hatch="//",  label=r'$\bf{Conjunction\ Subgroups}$'),#\ (\mathcal{S}_A)}$'),


    # Patch(facecolor=method_colors[methods[0]], hatch="\\", edgecolor='black', label=method_names[methods[0]]),
    Patch(facecolor=method_colors[methods[0]], hatch="//", label=method_names[methods[0]]),
    # Patch(facecolor=method_colors[methods[0]], edgecolor='black', label=method_names[methods[0]]),
    Patch(color='none', label=''),
    Patch(facecolor=method_colors[methods[1]], hatch="//", label=method_names[methods[1]]),
    # Patch(facecolor=method_colors[methods[1]], edgecolor='black', label=method_names[methods[1]]),
# ]
    Patch(color='none', label=''),
    Patch(color='none', label=''),

# legend_elements2 = [
    # Section 2 Header
    Patch(color='none', label=r'$\bf{Linear\ Model\ Subgroups}$'),#\ (\mathcal{S}_L)}$'),
    # Patch(facecolor=method_colors[methods[2]], hatch="\\", edgecolor='black', label=method_names[methods[2]]),
    Patch(facecolor=method_colors[methods[2]], label=method_names[methods[2]]),
    Patch(color='none', label=''),
    # Patch(facecolor=method_colors[methods[3]], hatch="x", edgecolor='black', label=method_names[methods[3]]),
    Patch(facecolor=method_colors[methods[3]], label=method_names[methods[3]]),
    Patch(color='none', label=''),
    # Patch(facecolor=method_colors[methods[4]], hatch="/", edgecolor='black', label=method_names[methods[4]]),
    Patch(facecolor=method_colors[methods[4]], label=method_names[methods[4]]),
]

# Place legend at the bottom
# fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.42, 0),
#            ncol=2, frameon=True, fontsize=10, borderpad=1, handletextpad=0.5, facecolor="#ACACAC")
# fig.legend(handles=legend_elements2, loc='lower left', bbox_to_anchor=(0.42, 0),
#            ncol=3, frameon=True, fontsize=10, borderpad=1, handletextpad=0.5)

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0),
           ncol=6, frameon=True, fontsize=10, borderpad=1, handletextpad=0.5, columnspacing=1.5 )


# fig.legend(handles, labels, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1))
# fig.legend(
    # handles, labels, loc="lower center", ncol=len(methods), bbox_to_anchor=(0.5, 0)
# )

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.tight_layout(rect=[0, 0.25, 1, 1])
output_path = (
    "plots/" + str(date.today()) + f"_eval.pdf"
)
plt.savefig(output_path, dpi=300)

plt.show()
