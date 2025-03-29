import numpy as np
import matplotlib.pyplot as plt

#* Data for SIMDRAM
our = [28800000, 3370000, 22700000, 12000000, 12000000, 22800000, 22700000, 22800000, 12000000, 22800000, 22700000, 12195410, 22785200, 22691540, 462700]
heuristic = [118000000, 3810000, 28300000, 24200000, 15200000, 90400000, 49200000, 90400000, 27200000, 30400000, 43200000, 27202710, 54401180, 31224630, 1253745]
pimdl = [37933616, 6224624, 25900000, 21765520, 13653472, 24372418, 33148752 ,27306944, 12444740, 22481032, 43218952, 12272386, 24372418, 23019920, 462700]
cosa = [31500000, 3770000, 27800000, 13800000, 12200000, 24400000, 26000000, 24400000, 13800000, 24400000, 24600000, 12195410, 22785200, 25204980, 462700]

x_labels = ['Conv0', 'Conv1', 'Conv2', 'Conv3', 'Conv5', 'Conv11', 'Conv12',
           'Conv14', 'Conv15', 'Conv24', 'Conv25',
           'Conv45', 'Conv46', 'Conv48', 'FC0']

method_list = ["OptiPIM", "Heur", "PIMDL", "ASIC"]

# Calculating speedup compared to the maximum value within each x label
methods_data = [our, heuristic, pimdl, cosa]
speedup_data = []

for i in range(len(x_labels)):
    max_value = max(our[i], heuristic[i], pimdl[i], cosa[i])
    speedup_data.append([max_value / our[i], max_value / heuristic[i], max_value / pimdl[i], max_value / cosa[i]])

speedup_data = np.array(speedup_data).T

#* Bert Data for SIMDRAM
bert_our = [14034020, 2617716, 2617716, 14034020, 55053648, 55053648]
bert_heuristic = [23504256, 7310464, 7310464, 23504256, 89298432, 89298432]
bert_pimdl = [14034024, 2617716, 2617716, 14034024, 55053648, 55053648]
bert_cosa = [15264480, 3724464, 3724464, 15264480, 60271490, 65776510]

x_labels_bert = ["Q/K/V_proj", "Attention:QxK", "Attention:AxV", "O_proj", "up_proj", "dn_proj"]

# Calculating speedup compared to the maximum value within each x label for Bert data
methods_data_bert = [bert_our, bert_heuristic, bert_pimdl, bert_cosa]
speedup_data_bert = []

for i in range(len(x_labels_bert)):
    max_value = max(bert_our[i], bert_heuristic[i], bert_pimdl[i], bert_cosa[i])
    speedup_data_bert.append([max_value / bert_our[i], max_value / bert_heuristic[i], max_value / bert_pimdl[i], max_value / bert_cosa[i]])

speedup_data_bert = np.array(speedup_data_bert).T

#* Data for HBM-PIM
our_hbm = [28800, 17100, 33400, 30720, 59300, 68400 ,37952, 48900, 37400, 49700, 35900, 30800, 49200, 40300, 7150]
heuristic_hbm = [7380000, 806000, 7230000, 3220000, 3220000, 6430000, 3834254, 6430000, 3230000, 6440000, 7230000, 3230000, 6470000, 7240000, 7150]
pimdl_hbm = [35924, 17100, 52608, 34304, 59456, 68480, 52320, 49152, 37376, 49664, 45312, 30848, 55424, 40256, 7150]
cosa_hbm = [1869190, 806000, 1831556, 1608956, 3220000, 3223902, 1810408, 1612280, 809276, 815736, 1809596, 3230000, 408992, 3619636, 7150]

# Calculating speedup compared to the maximum value within each x label for HBM-PIM data
methods_data_hbm = [our_hbm, heuristic_hbm, pimdl_hbm, cosa_hbm]
speedup_data_hbm = []

for i in range(len(x_labels)):
    max_value = max(our_hbm[i], heuristic_hbm[i], pimdl_hbm[i], cosa_hbm[i])
    speedup_data_hbm.append([max_value / our_hbm[i], max_value / heuristic_hbm[i], max_value / pimdl_hbm[i], max_value / cosa_hbm[i]])

speedup_data_hbm = np.array(speedup_data_hbm).T
speedup_data_hbm = np.where(speedup_data_hbm == 1, 0.1, np.log10(speedup_data_hbm))

#* Bert Data for HBM-PIM
bert_our_hbm = [39552, 13184, 13184, 39552, 97536, 97536]
bert_heuristic_hbm = [112464, 100216, 100216, 112464, 154944, 154944]
bert_pimdl_hbm = [39552, 13184, 16832, 39552, 97536, 97536]
bert_cosa_hbm = [1575324, 395034, 395034, 1575324, 799344, 3154032]

# Calculating speedup compared to the maximum value within each x label for Bert HBM-PIM data
methods_data_bert_hbm = [bert_our_hbm, bert_heuristic_hbm, bert_pimdl_hbm, bert_cosa_hbm]
speedup_data_bert_hbm = []

for i in range(len(x_labels_bert)):
    max_value = max(bert_our_hbm[i], bert_heuristic_hbm[i], bert_pimdl_hbm[i], bert_cosa_hbm[i])
    speedup_data_bert_hbm.append([max_value / bert_our_hbm[i], max_value / bert_heuristic_hbm[i], max_value / bert_pimdl_hbm[i], max_value / bert_cosa_hbm[i]])

speedup_data_bert_hbm = np.array(speedup_data_bert_hbm).T

# Plotting
x = np.arange(len(x_labels))  # the label locations for first plot
x_bert = np.arange(len(x_labels_bert))  # the label locations for second plot
width = 0.2  # the width of the bars

hatches = ['O', '..', '--', 'xx']
brightness = [1.0, 0.9, 0.85, 0.8]  # Different brightness levels for the bars
color_platte = ["#0000b3", "#0020ff", "#c9e52f", "#a6d75b"]

fig, axes = plt.subplots(2, 2, gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.08, 'hspace': 0.35}, figsize=(24, 8))
(ax1, ax2), (ax3, ax4) = axes

# Add dashed horizontal grid lines to all plots
for ax in [ax1, ax2, ax3, ax4]:
    ax.yaxis.grid(True, linestyle='--', which='both', color='grey', alpha=0.7)
ax2.yaxis.set_label_position('right')
ax4.yaxis.set_label_position('right')

# Plot bars for each method in the first plot
offset = -1.5 * width
counter = 0
for method, speedup in zip(method_list, speedup_data):
    # ax1.bar(x + offset, speedup, width, label=method, color=plt.cm.Set1(0.5 * (counter / len(method_list))), alpha=brightness[counter % len(brightness)], hatch=hatches[counter % len(hatches)], edgecolor='black')
    ax1.bar(x + offset, speedup, width, label=method, color=plt.cm.tab20c(4 * counter + counter), edgecolor='black')
    offset += width
    counter += 1

# Plot bars for each method in the second plot
offset = -1.5 * width
counter = 0
for method, speedup in zip(method_list, speedup_data_bert):
    # ax2.bar(x_bert + offset, speedup, width, color=plt.cm.Set1(0.5 * (counter / len(method_list))), alpha=brightness[counter % len(brightness)], hatch=hatches[counter % len(hatches)], edgecolor='black')
    ax2.bar(x_bert + offset, speedup, width, color=plt.cm.tab20c(4 * counter + counter), edgecolor='black')
    offset += width
    counter += 1

# Plot bars for each method in the third plot
offset = -1.5 * width
counter = 0
for method, speedup in zip(method_list, speedup_data_hbm):
    # ax3.bar(x + offset, speedup, width, color=plt.cm.Set1(0.5 * (counter / len(method_list))), alpha=brightness[counter % len(brightness)], hatch=hatches[counter % len(hatches)], edgecolor='black')
    ax3.bar(x + offset, speedup, width, color=plt.cm.tab20c(4 * counter + counter), edgecolor='black')
    offset += width
    counter += 1

# Plot bars for each method in the fourth plot
offset = -1.5 * width
counter = 0
for method, speedup in zip(method_list, speedup_data_bert_hbm):
    # ax4.bar(x_bert + offset, speedup, width, color=plt.cm.Set1(0.5 * (counter / len(method_list))), alpha=brightness[counter % len(brightness)], hatch=hatches[counter % len(hatches)], edgecolor='black')
    ax4.bar(x_bert + offset, speedup, width, color=plt.cm.tab20c(4 * counter + counter), edgecolor='black')
    offset += width
    counter += 1

# Labels and formatting for first plot
ax1.set_ylabel('Speed-up', fontsize=13, fontweight='bold')
ax1.set_xlabel('(a) ResNet-50 (SIMDRAM)', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, rotation=15, ha='right', fontsize=13)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=4, fontsize=13)

# Labels and formatting for second plot
# ax2.set_ylabel('Speedup', labelpad=20, rotation=270)
ax2.set_xlabel('(b) Bert (SIMDRAM)', fontsize=13, fontweight='bold')
ax2.set_xticks(x_bert)
ax2.set_xticklabels(x_labels_bert, rotation=15, ha='right', fontsize=13)

# Labels and formatting for third plot
ax3.set_ylabel('Speed-up', fontsize=13, fontweight='bold')
ax3.set_xlabel('(c) ResNet-50 (HBM-PIM)', fontsize=13, fontweight='bold')
ax3.set_yticks([0, 0.1, 1, 2])
ax3.set_yticklabels(['0', '$10^0$', '$10^1$', '$10^2$'])
ax3.set_xticks(x)
ax3.set_xticklabels(x_labels, rotation=15, ha='right', fontsize=13)

# Labels and formatting for fourth plot
# ax4.set_ylabel('Speedup', labelpad=20, rotation=270)
ax4.set_xlabel('(d) Bert (HBM-PIM)', fontsize=13, fontweight='bold')
ax4.set_xticks(x_bert)
ax4.set_xticklabels(x_labels_bert, rotation=15, ha='right', fontsize=13)

plt.subplots_adjust(wspace=0.4, bottom=0.1)
# plt.show()

plt.savefig('./Graphs/layer_breakdown.pdf', bbox_inches='tight', format="pdf")
