import matplotlib.pyplot as plt
import numpy as np

# HBM-Data
x_labels = ['UNet', 'VGG16']
method_list = ['Base Index + Base Estimate', 'Base Index + OptiPIM Estimate', 'OptiPIM Index + Base Estimate', 'OptiPIM Index + OptiPIM Estimate']

unet_hbm = [8829168, 1828896, 7538944, 1140000]
unet_simdram = [1713957518, 1672341878, 1710000000, 1670000000]
vgg_hbm = [6210144, 1779240.75, 6210000, 1779240.75]
vgg_simdram = [2944283484, 2820000000, 2940000000, 2820000000]

# Calculate speedup by dividing the maximum value in each category by each value
unet_hbm_speedup = [max(unet_hbm) / value for value in unet_hbm]
unet_simdram_speedup = [max(unet_simdram) / value for value in unet_simdram]
vgg_hbm_speedup = [max(vgg_hbm) / value for value in vgg_hbm]
vgg_simdram_speedup = [max(vgg_simdram) / value for value in vgg_simdram]

# Combine data for easier plotting
hbm_data = [unet_hbm_speedup, vgg_hbm_speedup]
simdram_data = [unet_simdram_speedup, vgg_simdram_speedup]

# Plotting
x = np.arange(len(x_labels))  # the label locations
width = 0.18  # the width of the bars, slightly increased to reduce spacing
spacing = 0.1  # spacing between bar groups

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'wspace': 0.1, 'bottom': 0.15})

# Plotting HBM results
for i in range(len(method_list)):
    ax1.bar(x + (i - 1.5) * (width + spacing / len(method_list)),
            [hbm_data[j][i] for j in range(len(x_labels))],
            width,
            label=method_list[i],
            color=plt.cm.tab20c(4 * i + i), edgecolor='black')

ax1.set_ylabel('Speedup', fontsize=18, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, fontsize=18)
ax1.set_xlabel('(a) HBM Results', fontsize=18, fontweight='bold')
ax1.yaxis.grid(True, linestyle='--', linewidth=0.7)

# Plotting SimDRAM results
for i in range(len(method_list)):
    ax2.bar(x + (i - 1.5) * (width + spacing / len(method_list)),
            [simdram_data[j][i] for j in range(len(x_labels))],
            width,
            label=method_list[i],
            color=plt.cm.tab20c(4 * i + i), edgecolor='black')

# ax2.set_ylabel('Speedup', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, fontsize=18)
ax2.set_xlabel('(b) SIMDRAM Results', fontsize=18, fontweight='bold')
ax2.set_ylim([0.8, 1.05])
ax2.yaxis.grid(True, linestyle='--', linewidth=0.7)

# Add a legend only once, instead of for both subplots
ax2.legend(loc='upper center', bbox_to_anchor=(-0.1, 1.2), ncol=2, fontsize=16)

# fig.tight_layout()
plt.savefig('./Graphs/optimization_comparison.pdf', bbox_inches='tight', format="pdf")
