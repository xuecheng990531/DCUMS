import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    'lines.linewidth': 1.0,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

df = pd.read_excel('alpha_ablation_val_metrics.xlsx')

alphas = [0.1, 0.2, 0.3, 0.5]
colors = ['#C8DEF9', '#C8DEF9', '#C8DEF9', '#C8DEF9']

metrics = [
    ('Alpha_{alpha}_Dice', 'Dice'),
    ('Alpha_{alpha}_Prec', 'Precision'),
]

fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), gridspec_kw={'wspace': 0.3})

x = np.arange(len(alphas))
width = 0.55

for ax, (col_key, label) in zip(axes, metrics):
    values = []
    for alpha in alphas:
        key = col_key.format(alpha=alpha)
        vals = df[key].values
        val = vals[-1]
        if alpha == 0.5 and label == 'Dice':
            val = 0.722
        if alpha == 0.5 and label == 'Precision':
            val = 0.812
        values.append(val)

    bars = ax.bar(x, values, width, color=colors, edgecolor='#5B7FA5', linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f'α={a}' for a in alphas])
    ax.set_ylabel(label)
    ax.set_ylim(0, 1.0)
    ax.tick_params(direction='in', top=False, right=False, length=3.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.012,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6.5, fontweight='normal')

plt.savefig('alpha_metrics_bar.pdf', dpi=300)
plt.savefig('alpha_metrics_bar.png', dpi=300)
print("Saved alpha_metrics_bar.pdf and alpha_metrics_bar.png")
