import re
import numpy as np
import pandas as pd

def extract_train_losses(filepath):
    losses = []
    iterations = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'Iter (\d+)/\d+, Train Loss: ([\d.]+)', line)
            if match:
                iterations.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return iterations, losses

def smooth_loss(losses, window=20):
    smoothed = []
    for i in range(len(losses)):
        start = max(0, i - window // 2)
        end = min(len(losses), i + window // 2 + 1)
        smoothed.append(np.mean(losses[start:end]))
    return smoothed

alphas = [0.1, 0.2, 0.3, 0.5]
all_data = {'Iteration': []}

max_len = 0
for alpha in alphas:
    log_path = f'experiment_logs/alpha_ablation_noal/alpha{alpha}/training_output.txt'
    iterations, losses = extract_train_losses(log_path)
    smoothed = smooth_loss(losses, window=20)
    all_data[f'Alpha_{alpha}_Raw'] = losses
    all_data[f'Alpha_{alpha}_Smoothed'] = smoothed
    max_len = max(max_len, len(losses))

max_len = min(len(all_data[f'Alpha_{alphas[0]}_Raw']), 2000)
for key in all_data:
    if key != 'Iteration':
        all_data[key] = all_data[key][:max_len]

all_data['Iteration'] = list(range(1, max_len + 1))

df = pd.DataFrame(all_data)

df.to_excel('alpha_ablation_smoothed_losses.xlsx', index=False)
print("Saved alpha_ablation_smoothed_losses.xlsx")
