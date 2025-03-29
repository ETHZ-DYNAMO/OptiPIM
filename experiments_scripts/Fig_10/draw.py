import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
#  Basic Config
# ------------------------------------------------------------------------------
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_PATH, "..", "..")  # or "/your/path"

x_labels    = ['AlexNet', 'ResNet50', 'ResNet152', 'UNet', 'VGG16', 'Bert', 'Stencil']
nn_models   = ["alexnet", "resnet50", "resnet152", "unet", "vgg16", "bert"]  # row indices 0..5
method_list = ["OptiPIM", "Heur", "PIMDL", "ASIC"]  # columns 0..3

NUM_MODELS  = len(x_labels)   # 7
NUM_METHODS = len(method_list)

input_loading_data    = np.zeros((NUM_MODELS, NUM_METHODS))
compute_data          = np.zeros((NUM_MODELS, NUM_METHODS))
output_reduction_data = np.zeros((NUM_MODELS, NUM_METHODS))

input_loading_data_2    = np.zeros((NUM_MODELS, NUM_METHODS))
compute_data_2          = np.zeros((NUM_MODELS, NUM_METHODS))
output_reduction_data_2 = np.zeros((NUM_MODELS, NUM_METHODS))

model_to_row = { name: i for (i, name) in enumerate(nn_models) }

# Baseline logs use a different capitalization
baseline_nn_mapping = {
    "alexnet":   "Alexnet",
    "resnet50":  "ResNet50",
    "resnet152": "resnet152",
    "unet":      "unet",
    "vgg16":     "vgg16",
    "bert":      "bert"
}

selected_heuristic = "heuristic_1" 

# ------------------------------------------------------------------------------
# CSV Parsing for columns 0 (OptiPIM), 1 (Heur), 3 (ASIC)
# ------------------------------------------------------------------------------
def read_optipim_breakdown_sum(csv_path):
    """Return {'our_input','our_compute','our_output'} from the 'sum' row (ignore extras)."""
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, 'r') as f:
        rd = csv.reader(f)
        header = next(rd, None)
        for row in rd:
            if len(row) >= 4 and row[0].strip().lower() == "sum":
                try:
                    return {
                        'our_input':   float(row[1]),
                        'our_compute': float(row[2]),
                        'our_output':  float(row[3])
                    }
                except:
                    return None
    return None

def read_cosa_breakdown_sum(csv_path):
    """Return {'cosa_input','cosa_compute','cosa_output'} from the 'sum' row."""
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, 'r') as f:
        rd = csv.reader(f)
        header = next(rd, None)
        for row in rd:
            if len(row) >= 4 and row[0].strip().lower() == "sum":
                try:
                    return {
                        'cosa_input':   float(row[1]),
                        'cosa_compute': float(row[2]),
                        'cosa_output':  float(row[3])
                    }
                except:
                    return None
    return None


def read_baseline_breakdown_sum(csv_path, model):
    if not os.path.isfile(csv_path):
        return (0,0,0)
    
    with open(csv_path, 'r') as f:
        rd = csv.reader(f)
        header = next(rd, None)  # read header

        rows = []
        for row in rd:
            if len(row) < 3:
                continue
            rows.append(row)

    if model != "bert":
        if len(rows) == 0:
            return (0,0,0)
        sum_data = rows[-1]
        try:
            return (float(sum_data[0]), float(sum_data[1]), float(sum_data[2]))
        except:
            return (0,0,0)
    else:
        custom_indices = [0, 1, 1, 3, 4, 5]
        in_sum = 0.0
        comp_sum = 0.0
        out_sum = 0.0
        for idx in custom_indices:
            if idx < 0 or idx >= len(rows):
                # If the CSV doesn't have enough rows, skip or break
                continue
            try:
                in_sum   += float(rows[idx][0])
                comp_sum += float(rows[idx][1])
                out_sum  += float(rows[idx][2])
            except:
                pass
        return (in_sum, comp_sum, out_sum)


def fill_arrays_for_device(device_type, device_baseline_folder, arr_in, arr_comp, arr_out):
    """
    For rows 0..5 => read from CSV for col0(OptiPIM), col1(Heur), col3(ASIC).
    """
    nb_optipim   = "32banks"
    nb_baseline  = "32_banks"
    cosa_csvname = "32banksbreak_down.csv"

    for model in nn_models:  # (alexnet..bert)
        row_idx = model_to_row[model]
        
        # (A) OptiPIM => col=0
        opti_csv = os.path.join(
            PROJECT_ROOT, "exp_results", "fig10", "final_results",
            "optipim", device_type, model, f"{nb_optipim}_breakdown.csv"
        )
        opti_data = read_optipim_breakdown_sum(opti_csv)
        if opti_data:
            arr_in[row_idx, 0]    = opti_data['our_input']
            arr_comp[row_idx, 0]  = opti_data['our_compute']
            arr_out[row_idx, 0]   = opti_data['our_output']

        # (B) Baseline => col=1
        baseline_csv = os.path.join(
            PROJECT_ROOT, "exp_results", "fig10", "final_results", "baseline",
            selected_heuristic, 
            baseline_nn_mapping[model],
            device_baseline_folder, f"{nb_baseline}_breakdown.csv"
        )
        (inL, cp, ouR) = read_baseline_breakdown_sum(baseline_csv, model)
        arr_in[row_idx, 1]   = inL
        arr_comp[row_idx, 1] = cp
        arr_out[row_idx, 1]  = ouR

        # (C) COSA => col=3
        cosa_csv = os.path.join(
            PROJECT_ROOT, "exp_results", "fig10", "final_results",
            "cosa", device_type, model, cosa_csvname
        )
        cosa_data = read_cosa_breakdown_sum(cosa_csv)
        if cosa_data:
            arr_in[row_idx, 3]   = cosa_data['cosa_input']
            arr_comp[row_idx, 3] = cosa_data['cosa_compute']
            arr_out[row_idx, 3]  = cosa_data['cosa_output']

# ------------------------------------------------------------------------------
#  Fill columns 0,1,3 from CSV for the 6 CNN/RNN models
# ------------------------------------------------------------------------------
fill_arrays_for_device("PUM", "simdram_baseline", input_loading_data, compute_data, output_reduction_data)
fill_arrays_for_device("NBP", "hbm_baseline",    input_loading_data_2, compute_data_2, output_reduction_data_2)

# ------------------------------------------------------------------------------
# 2) The Results for PIMDL. Not key results in our paper, 
#    so we directly provide
# ------------------------------------------------------------------------------
pimdl_col_simdram = {
    0: (9430440,     191640800,  15988996),   # AlexNet
    1: (53421312,    1546935525, 77585920),   # ResNet50
    2: (128629696,   2857597075, 216761600),  # ResNet152
    3: (15653120,    5418159575, 480598272),  # UNet
    4: (8927488,     6695778900, 675293696),  # VGG16
    5: (2396160,     139029525,  4337664),    # Bert
}
pimdl_col_hbm = {
    0: (119600.5625, 251154,     45232),      # AlexNet
    1: (1013140,     724483,     438618),     # ResNet50
    2: (2378164,     2323911,    1306618),    # ResNet152
    3: (5646258,     823488,     433776),     # UNet
    4: (781677.5,    6381508,    200052),     # VGG16
    5: (86784,       120144,      96352),      # Bert
}

for i in range(6):
    # SIMDRAM
    (inL, cp, ouR) = pimdl_col_simdram[i]
    input_loading_data[i, 2]    = inL
    compute_data[i, 2]          = cp
    output_reduction_data[i, 2] = ouR
    
    # HBM-PIM
    (inL2, cp2, ouR2) = pimdl_col_hbm[i]
    input_loading_data_2[i, 2]    = inL2
    compute_data_2[i, 2]          = cp2
    output_reduction_data_2[i, 2] = ouR2

# ------------------------------------------------------------------------------
# 3) The Results for Stencil workload. Not key results in our paper, 
#    so we directly provide
# ------------------------------------------------------------------------------
stencil_idx = 6

# SIMDRAM:
input_loading_data[stencil_idx]    = [52080,  89280,  52080,  89280]
compute_data[stencil_idx]          = [558150, 841275, 558150, 554100]
output_reduction_data[stencil_idx] = [14880,  4960,   14880,  29760]

# HBM-PIM:
input_loading_data_2[stencil_idx]    = [1134,   89280, 2424,  89280]
compute_data_2[stencil_idx]          = [960,    13248, 72,    24]
output_reduction_data_2[stencil_idx] = [58,     1,     77.5,  620]

# ------------------------------------------------------------------------------
# Finish Data Extraction
# ------------------------------------------------------------------------------
print("[INFO] Data Extraction Finished for all models and methods")
print("[INFO] Drawing Fig 10.")

# ------------------------------------------------------------------------------
#                               Plotting   
# ------------------------------------------------------------------------------

# ----------------------------------------------------------------
# 2. Speed-up Calculations for SIMDRAM (subplot a)
# ----------------------------------------------------------------

# Summing the input_loading, compute, and output_reduction data for SIMDRAM
summed_data = []
for i in range(len(x_labels)):
    nn_model_sum = [
        input_loading_data[i][j] + compute_data[i][j] + output_reduction_data[i][j]
        for j in range(len(method_list))
    ]
    summed_data.append(nn_model_sum)

# Calculate speed-up for each method in each neural network model
speedup_data = []
for data in summed_data:
    max_value = max(data)
    speedup_data.append([max_value / val for val in data])

# ----------------------------------------------------------------
# 3. Speed-up Calculations for HBM-PIM (subplot c)
# ----------------------------------------------------------------

# Summing the input_loading, compute, and output_reduction data for HBM-PIM
summed_data_2 = []
for i in range(len(x_labels)):
    nn_model_sum = [
        input_loading_data_2[i][j] + compute_data_2[i][j] + output_reduction_data_2[i][j]
        for j in range(len(method_list))
    ]
    summed_data_2.append(nn_model_sum)

# Calculate speed-up for each method in each neural network model
speedup_data_2 = []
for data in summed_data_2:
    max_value = max(data)
    speedup_data_2.append([max_value / val for val in data])

# ----------------------------------------------------------------
# 4. Create the Figure
# ----------------------------------------------------------------
fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(
    2, 2, figsize=(12, 8),
    gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.1, 'hspace': 0.25}
)

# ----------------------------------------------------------------
# 5. Adjust the Spacing Between Groups and the Bar Width
#    (get extra spacing and smaller bars)
# ----------------------------------------------------------------
group_spacing = 1.1  # Controls how spread out each group is
width = 0.18         # Shrink the bar width

# Create new x positions that incorporate the group spacing
x = np.arange(len(x_labels)) * group_spacing

# ----------------------------------------------------------------
# 6. Subplot (a): Bar Chart for SIMDRAM Speedup
# ----------------------------------------------------------------
for i, method in enumerate(method_list):
    ax.bar(
        x + i * width,
        [speedup_data[j][i] for j in range(len(x_labels))],
        width,
        label=method,
        color=plt.cm.tab20c(4 * i + i),
        edgecolor='black'
    )

# Add horizontal dashed lines
ymax = max(max(row) for row in speedup_data)
for yv in np.arange(1.0, ymax, 0.5):
    ax.axhline(y=yv, color='gray', linestyle='--', linewidth=0.5)

ax.set_xlabel('(a) Results for SIMDRAM', fontsize=13, fontweight='bold')
ax.set_ylabel('Speed-up', fontsize=13, fontweight='bold')

# Set x ticks for the grouped bars
ax.set_xticks(x + width * (len(method_list) / 2 - 0.5))
ax.set_xticklabels(x_labels, fontsize=13)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.1),
    fancybox=False,
    shadow=False,
    ncol=len(method_list),
    framealpha=0.3,
    borderpad=0.1,
    fontsize=13
)

# ----------------------------------------------------------------
# 7. Subplot (b): Stacked Bar Chart (SIMDRAM Breakdown)
# ----------------------------------------------------------------
input_sum = [sum([input_loading_data[i][j] for i in range(len(x_labels))]) for j in range(len(method_list))]
compute_sum = [sum([compute_data[i][j] for i in range(len(x_labels))]) for j in range(len(method_list))]
output_sum = [sum([output_reduction_data[i][j] for i in range(len(x_labels))]) for j in range(len(method_list))]
total_sum = [input_sum[i] + compute_sum[i] + output_sum[i] for i in range(len(method_list))]

input_normalized = [input_sum[i] / total_sum[i] for i in range(len(method_list))]
compute_normalized = [compute_sum[i] / total_sum[i] for i in range(len(method_list))]
output_normalized = [output_sum[i] / total_sum[i] for i in range(len(method_list))]

for i, method in enumerate(method_list):
    ax2.bar(
        method,
        compute_normalized[i],
        width=0.5,
        label='Compute' if i == 0 else "",
        color='#1984c5'
    )
    ax2.bar(
        method,
        input_normalized[i],
        width=0.5,
        bottom=compute_normalized[i],
        label='Input Loading' if i == 0 else "",
        color='#a6d75b'
    )
    ax2.bar(
        method,
        output_normalized[i],
        width=0.5,
        bottom=np.array(compute_normalized[i]) + np.array(input_normalized[i]),
        label='Output Reduction' if i == 0 else "",
        color=plt.cm.tab20c(5)
    )

ax2.set_xlabel('(b) SIMDRAM Perf. Breakdown', fontsize=13, fontweight='bold')
ax2.set_ylabel('Normalized Contribution', fontsize=13, fontweight='bold')
ax2.set_xticklabels(method_list, fontsize=13)
ax2.yaxis.set_label_position('right')
ax2.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.2),
    ncol=2,
    fancybox=False,
    shadow=False,
    framealpha=0.3,
    borderpad=0.1,
    fontsize=13,
    handletextpad=0.2,
    columnspacing=0.1
)

# ----------------------------------------------------------------
# 8. Subplot (c): Bar Chart for HBM-PIM Speedup
# ----------------------------------------------------------------
for i, method in enumerate(method_list):
    ax3.bar(
        x + i * width,
        [
            0.1 if speedup_data_2[j][i] == 1 else np.log10(speedup_data_2[j][i])
            for j in range(len(x_labels))
        ],
        width,
        label=method,
        color=plt.cm.tab20c(4 * i + i),
        edgecolor='black'
    )

# Add horizontal dashed lines for 0.1, 1, 2 in log-scale
for yv in [0.1, 1, 2]:
    ax3.axhline(y=yv, color='gray', linestyle='--', linewidth=0.5)

ax3.set_yticks([0.1, 1, 2])
ax3.set_yticklabels([r'$10^0$', r'$10^1$', r'$10^2$'])

ax3.set_xlabel('(c) Results for HBM-PIM', fontsize=13, fontweight='bold')
ax3.set_ylabel('Speed-up', fontsize=13, fontweight='bold')

ax3.set_xticks(x + width * (len(method_list) / 2 - 0.5))
ax3.set_xticklabels(x_labels, fontsize=13)

# ----------------------------------------------------------------
# 9. Subplot (d): Stacked Bar Chart (HBM-PIM Breakdown)
# ----------------------------------------------------------------
input_sum_2 = [sum([input_loading_data_2[i][j] for i in range(len(x_labels))]) for j in range(len(method_list))]
compute_sum_2 = [sum([compute_data_2[i][j] for i in range(len(x_labels))]) for j in range(len(method_list))]
output_sum_2 = [sum([output_reduction_data_2[i][j] for i in range(len(x_labels))]) for j in range(len(method_list))]
total_sum_2 = [input_sum_2[i] + compute_sum_2[i] + output_sum_2[i] for i in range(len(method_list))]

input_normalized_2 = [input_sum_2[i] / total_sum_2[i] for i in range(len(method_list))]
compute_normalized_2 = [compute_sum_2[i] / total_sum_2[i] for i in range(len(method_list))]
output_normalized_2 = [output_sum_2[i] / total_sum_2[i] for i in range(len(method_list))]

for i, method in enumerate(method_list):
    ax4.bar(
        method,
        compute_normalized_2[i],
        width=0.5,
        label='Compute' if i == 0 else "",
        color='#1984c5'
    )
    ax4.bar(
        method,
        input_normalized_2[i],
        width=0.5,
        bottom=compute_normalized_2[i],
        label='Input Loading' if i == 0 else "",
        color='#a6d75b'
    )
    ax4.bar(
        method,
        output_normalized_2[i],
        width=0.5,
        bottom=np.array(compute_normalized_2[i]) + np.array(input_normalized_2[i]),
        label='Output Reduction' if i == 0 else "",
        color=plt.cm.tab20c(5)
    )

ax4.set_ylabel('Normalized Contribution', fontsize=13, fontweight='bold')
ax4.set_xlabel('(d) HBM-PIM Perf. Breakdown', fontsize=13, fontweight='bold')
ax4.set_xticklabels(method_list, fontsize=13)
ax4.yaxis.set_label_position('right')

plt.subplots_adjust(wspace=0.2, hspace=0.05)

# Check the existence of the folder
if not os.path.exists(f"{PROJECT_ROOT}/exp_results/fig10"):
    os.makedirs(f"{PROJECT_ROOT}/exp_results/fig10")

# Save the figure
plt.savefig(f"{PROJECT_ROOT}/exp_results/fig10/fig10.pdf", bbox_inches='tight', format="pdf")

