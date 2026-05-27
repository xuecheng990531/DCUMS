import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Nature 风格设置
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 7,
    'axes.linewidth': 0.8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'lines.linewidth': 1.2,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# 读取数据
df = pd.read_excel('alpha_ablation_val_metrics.xlsx')

alphas = [0.1, 0.2, 0.3, 0.5]
colors = ['#D55E00', '#0072B2', '#009E73', '#CC79A7']
line_styles = ['-', '--', '-.', ':']
iterations = df['Iteration'].values

fig, ax = plt.subplots(figsize=(3.5, 2.8))

for alpha, color, ls in zip(alphas, colors, line_styles):
    smoothed = df[f'Alpha_{alpha}_ValLoss_Smoothed'].values
    ax.plot(iterations, smoothed, color=color, linestyle=ls, linewidth=1.2,
            label=f'α = {alpha}')

ax.set_xlabel('Iteration', fontsize=8, fontweight='normal')
ax.set_ylabel('Validation Loss', fontsize=8, fontweight='normal')
ax.legend(frameon=False, loc='upper right', fontsize=6.5)

ax.set_xlim(0, 2050)
ax.set_ylim(0.4, 1.05)
ax.set_xticks([0, 400, 800, 1200, 1600, 2000])
ax.tick_params(direction='in', top=True, right=True, length=3.5)

# 添加 inset（最后 100 轮放大，即迭代 1600-2000）
axins = inset_axes(ax, width="45%", height="45%", loc='lower left',
                   borderpad=1.5)

start_idx = iterations >= 1600
x_zoom = iterations[start_idx]

for alpha, color, ls in zip(alphas, colors, line_styles):
    smoothed = df[f'Alpha_{alpha}_ValLoss_Smoothed'].values[start_idx]
    axins.plot(x_zoom, smoothed, color=color, linestyle=ls, linewidth=1.2)

axins.set_xlim(1600, 2000)
axins.set_ylim(0.48, 0.65)
axins.set_xticks([1600, 1800, 2000])
axins.set_xticklabels(['1600', '1800', '2000'], fontsize=6)
axins.set_yticks([0.50, 0.55, 0.60, 0.65])
axins.set_yticklabels(['0.50', '0.55', '0.60', '0.65'], fontsize=6)
axins.tick_params(direction='in', top=True, right=True, length=2.5)
axins.spines['top'].set_linewidth(0.5)
axins.spines['right'].set_linewidth(0.5)
axins.spines['bottom'].set_linewidth(0.5)
axins.spines['left'].set_linewidth(0.5)

# 添加矩形框指示 zoom 区域
rect = plt.Rectangle((1600, 0.48), 400, 0.17, linewidth=0.8,
                     edgecolor='black', facecolor='none', zorder=5)
ax.add_patch(rect)

# 连接矩形框到 inset 的四个角
con1 = ConnectionPatch(xyA=(1600, 0.48), xyB=(1600, 0.48),
                       coordsA="data", coordsB="data",
                       axesA=ax, axesB=axins, color='gray', linewidth=0.5)
con2 = ConnectionPatch(xyA=(2000, 0.48), xyB=(2000, 0.48),
                       coordsA="data", coordsB="data",
                       axesA=ax, axesB=axins, color='gray', linewidth=0.5)
con3 = ConnectionPatch(xyA=(2000, 0.65), xyB=(2000, 0.65),
                       coordsA="data", coordsB="data",
                       axesA=ax, axesB=axins, color='gray', linewidth=0.5)

ax.add_artist(con1)
ax.add_artist(con2)
ax.add_artist(con3)

plt.tight_layout(pad=0.2)
plt.savefig('alpha_loss_smoothed.pdf', dpi=300)
plt.savefig('alpha_loss_smoothed.png', dpi=300)
print("Saved alpha_loss_smoothed.pdf and alpha_loss_smoothed.png")
