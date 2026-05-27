import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_metrics_from_log(filepath):
    iterations = []
    val_losses = []
    val_ious = []
    val_dices = []
    val_precs = []
    val_recs = []
    train_losses = []

    with open(filepath, 'r') as f:
        for line in f:
            val_match = re.search(r'\[Iter (\d+)\] Val Loss: ([\d.]+), IoU: ([\d.]+), Dice: ([\d.]+), Prec: ([\d.]+), Rec: ([\d.]+)', line)
            if val_match:
                iterations.append(int(val_match.group(1)))
                val_losses.append(float(val_match.group(2)))
                val_ious.append(float(val_match.group(3)))
                val_dices.append(float(val_match.group(4)))
                val_precs.append(float(val_match.group(5)))
                val_recs.append(float(val_match.group(6)))

            train_match = re.search(r'Iter \d+/\d+, Train Loss: ([\d.]+)', line)
            if train_match:
                train_losses.append(float(train_match.group(1)))

    return {
        'iterations': iterations,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'val_dices': val_dices,
        'val_precs': val_precs,
        'val_recs': val_recs,
        'train_losses': train_losses,
        'best_iou': max(val_ious) if val_ious else 0,
        'best_dice': val_dices[val_ious.index(max(val_ious))] if val_ious else 0,
        'best_prec': val_precs[val_ious.index(max(val_ious))] if val_ious else 0,
        'best_rec': val_recs[val_ious.index(max(val_ious))] if val_ious else 0,
    }


alphas = [0.1, 0.2, 0.3, 0.5]
results = {}

for alpha in alphas:
    log_path = f'experiment_logs/alpha_ablation_noal/alpha{alpha}/training_output.txt'
    try:
        results[alpha] = extract_metrics_from_log(log_path)
        print(f"alpha={alpha}: loaded {len(results[alpha]['iterations'])} validation points")
    except Exception as e:
        print(f"alpha={alpha}: error - {e}")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

ax = axes[0, 0]
for alpha, color in zip(alphas, colors):
    if alpha in results:
        ax.plot(results[alpha]['iterations'], results[alpha]['val_losses'], 
                color=color, linewidth=2, label=f'alpha={alpha}')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for alpha, color in zip(alphas, colors):
    if alpha in results:
        ax.plot(results[alpha]['iterations'], results[alpha]['val_ious'], 
                color=color, linewidth=2, label=f'alpha={alpha}')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Validation IoU', fontsize=12)
ax.set_title('Validation IoU Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for alpha, color in zip(alphas, colors):
    if alpha in results:
        ax.plot(results[alpha]['iterations'], results[alpha]['val_dices'], 
                color=color, linewidth=2, label=f'alpha={alpha}')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Validation Dice', fontsize=12)
ax.set_title('Validation Dice Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for alpha, color in zip(alphas, colors):
    if alpha in results:
        steps = np.arange(len(results[alpha]['train_losses'])) * 40
        sample_losses = results[alpha]['train_losses'][::10]
        sample_steps = steps[::10]
        ax.plot(sample_steps, sample_losses, color=color, linewidth=1.5, 
                label=f'alpha={alpha}', alpha=0.8)
ax.set_xlabel('Training Iteration', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('alpha_loss_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved alpha_loss_curves.png")


metrics = ['val_ious', 'val_dices', 'val_precs', 'val_recs']
metric_names = ['IoU', 'Dice', 'Precision', 'Recall']
best_values = {m: [] for m in metrics}

for alpha in alphas:
    if alpha in results:
        r = results[alpha]
        best_idx = r['iterations'].index(max(r['iterations']))
        for m in metrics:
            best_values[m].append(r[m][best_idx] if r[m] else 0)


fig, axes = plt.subplots(1, 4, figsize=(16, 5))
x = np.arange(len(alphas))
width = 0.5

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx]
    bars = ax.bar(x, best_values[metric], width, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'α={a}' for a in alphas], fontsize=12)
    ax.set_ylabel(name, fontsize=12)
    ax.set_title(f'Best {name}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, best_values[metric]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('alpha_metrics_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved alpha_metrics_bar.png")
