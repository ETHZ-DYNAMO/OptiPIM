from __future__ import annotations
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os
import json

RESULTS_PATH = "exp_results/fig14"

def parse_results_json(json_file: str):
    # The fixed ordering you want
    X_LABELS = ["alexnet", "resnet50", "resnet152", "unet", "vgg16", "bert"]
    BANK_SETTING_LIST = [
        (256, False),  # 8GB-Even
        (256, True),   # 8GB-OptiPIM
        (512, False),  # 16GB-Even
        (512, True),   # 16GB-OptiPIM
        (1024, False), # 32GB-Even
        (1024, True),  # 32GB-OptiPIM
    ]
    
    # Load JSON
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Convert to dict keyed by (model, arch, n_banks, grouping)
    # for quick lookup: (str, str, int, bool) -> cycles
    results_dict = {}
    for entry in data:
        key = (entry["model"], entry["arch"], entry["n_banks"], entry["grouping"])
        results_dict[key] = entry["cycles"]
    
    # Build the final structures
    SIMDRAM_COST = []
    HBM_COST = []
    
    # For each model in X_LABELS order, 
    # build a row in simdram_cost and hbm_cost of length 6 
    # following the bank-setting pairs in BANK_SETTING_LIST
    for model in X_LABELS:
        simdram_row = []
        hbm_row = []
        
        for (n_banks, grouping) in BANK_SETTING_LIST:
            # Lookup cycles from the dictionary
            simdram_val = results_dict.get((model, "simdram", n_banks, grouping), 0)
            hbm_val     = results_dict.get((model, "hbm",     n_banks, grouping), 0)
            
            simdram_row.append(simdram_val)
            hbm_row.append(hbm_val)
        
        SIMDRAM_COST.append(simdram_row)
        HBM_COST.append(hbm_row)
    
    # Return them in a convenient dictionary or just print them
    return {
        "X_LABELS": ["AlexNet", "ResNet-50", "ResNet-152", "UNet", "VGG16", "Bert"],
        "BANK_SETTING_LIST": [
            "8GB-Even",
            "8GB-OptiPIM",
            "16GB-Even",
            "16GB-OptiPIM",
            "32GB-Even",
            "32GB-OptiPIM"
        ],
        "SIMDRAM_COST": SIMDRAM_COST,
        "HBM_COST": HBM_COST
    }


def plot_arch(plot_data: Dict[str, Any] ,arch_data: List[List[int]], ax, legend: int, title: str, interval: float, x_height: float, lower_bound: float | None = None, upper_bound: float | None = None,):
    # Calculating speed-up for each method in each neural network model
    speedup_data: List[List[float]] = []
    for model_data in arch_data:
        if model_data[0] == -1:
            base_value = model_data[1]
        else:
            base_value = model_data[0]
        model_speedup = [(base_value / val) if val != -1 else 0 for val in model_data]
        # from rich import print
        # print(model_speedup)
        speedup_data.append(model_speedup)

    # Plotting the bar graph
    x = np.arange(len(plot_data["X_LABELS"]))  # the label locations
    width = 0.15  # the width of the bars

    if lower_bound is not None:
        ax.set_ylim(lower_bound, upper_bound)

    # Plot each method as a separate set of bars on the left plot
    for i, method in enumerate(plot_data["BANK_SETTING_LIST"]):
        ax.bar(
            x + i * width,
            [speedup_data[j][i] for j in range(len(plot_data["X_LABELS"]))],
            width,
            label=method,
            color=plt.cm.tab20c(1 * (i / len(plot_data["BANK_SETTING_LIST"]))), edgecolor='black'# type: ignore
        )

    # Put an X on the zero value
    for (j, model_speedup) in enumerate(speedup_data):
        for (i, speedup) in enumerate(model_speedup):
            if speedup == 0:
                ax.text(
                    x[j] + i * width,
                    x_height,
                    "X",
                    ha="center",
                    va="center",
                    color=plt.cm.Set1(1 * (i / len(plot_data["BANK_SETTING_LIST"]))), # type: ignore
                    fontsize=18,
                    fontweight="bold",
                )

    # Add horizontal dashed lines for the values on the y-axis
    if upper_bound is not None:
        upper_line = max(max(max(speedup_data)), upper_bound)
    else:
        upper_line = max(max(speedup_data))
    for y in np.arange(0.5, upper_line, interval):
        ax.axhline(y=y, color="gray", linestyle="--", linewidth=0.5)

    # Add some text for labels, title and custom x-axis tick labels, etc. for the left plot
    ax.set_xlabel(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Speed-up", fontsize=13, fontweight="bold")
    # fig.text(0.08, 0.5, 'Speed-up', va='center', rotation='vertical', fontsize=15)
    ax.set_xticks(x + width * (len(plot_data["BANK_SETTING_LIST"]) / 2 - 0.5))
    ax.set_xticklabels(plot_data["X_LABELS"], fontsize=13)
    if (legend):
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.22, 1.015),
            fancybox=False,
            shadow=False,
            ncol=len(plot_data["BANK_SETTING_LIST"])/3,
            framealpha=0.3,
            borderpad=0.1,
            fontsize=13,
        )

def main():
    fig, ((ax1), (ax2)) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"wspace": 0.1, "hspace": 0.2},
    )
    result_file = os.path.join(RESULTS_PATH, "holistic_results.json")
    plot_data = parse_results_json(result_file)
    plot_arch(plot_data, plot_data["SIMDRAM_COST"], ax1, 1, "(a) Results for SIMDRAM", 0.5, 0.25)
    plot_arch(plot_data, plot_data["HBM_COST"], ax2, 0, "(b) Results for HBM-PIM", 0.1, 0.83, 0.8, 1.59)

    # finialize the plot
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    # plt.show()

    # Save the fig
    fig_path = os.path.join(RESULTS_PATH, "fig14.pdf")
    plt.savefig(fig_path, bbox_inches="tight", format="pdf")

if __name__ == "__main__":
    main()