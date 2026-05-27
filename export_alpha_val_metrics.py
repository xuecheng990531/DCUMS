import re
import numpy as np
import pandas as pd

def extract_val_metrics(filepath):
    iterations = []
    val_losses = []
    val_ious = []
    val_dices = []
    val_precs = []
    val_recs = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'\[Iter (\d+)\] Val Loss: ([\d.]+), IoU: ([\d.]+), Dice: ([\d.]+), Prec: ([\d.]+), Rec: ([\d.]+)', line)
            if match:
                iterations.append(int(match.group(1)))
                val_losses.append(float(match.group(2)))
                val_ious.append(float(match.group(3)))
                val_dices.append(float(match.group(4)))
                val_precs.append(float(match.group(5)))
                val_recs.append(float(match.group(6)))
    return iterations, val_losses, val_ious, val_dices, val_precs, val_recs

def smooth_data(values, window=5):
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(np.mean(values[start:end]))
    return smoothed

alphas = [0.1, 0.2, 0.3, 0.5]
all_data = {}

for alpha in alphas:
    log_path = f'experiment_logs/alpha_ablation_noal/alpha{alpha}/training_output.txt'
    iters, losses, ious, dices, precs, recs = extract_val_metrics(log_path)
    all_data['Iteration'] = iters
    all_data[f'Alpha_{alpha}_ValLoss'] = losses
    all_data[f'Alpha_{alpha}_ValLoss_Smoothed'] = smooth_data(losses, window=3)
    all_data[f'Alpha_{alpha}_IoU'] = ious
    all_data[f'Alpha_{alpha}_Dice'] = dices
    all_data[f'Alpha_{alpha}_Prec'] = precs
    all_data[f'Alpha_{alpha}_Rec'] = recs

df = pd.DataFrame(all_data)
df.to_excel('alpha_ablation_val_metrics.xlsx', index=False)
print(f"Saved alpha_ablation_val_metrics.xlsx")
print(f"Columns: {list(df.columns)}")
print(f"Rows: {len(df)}")
print(df.head())
