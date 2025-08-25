import re
import glob
import pandas as pd

pattern = re.compile(r"^quan_(\w+)_(\d+)_([\d\.]+)\.csv$")

csv_files = glob.glob("*.csv")

groups = {}
for file in csv_files:
    m = pattern.match(file)
    if m:
        dataset, seed, quantile = m.groups()
        seed = int(seed) - 1
        key = (dataset, seed)
        if key not in groups:
            groups[key] = {}
        groups[key][quantile] = file

results = {}

for (dataset, seed), files in groups.items():
    if "0.05" in files and "0.95" in files:
        lower_file = files["0.05"]
        upper_file = files["0.95"]

        df_lower = pd.read_csv(lower_file)
        df_upper = pd.read_csv(upper_file)

        labels = df_lower["label"]
        lower_scores = df_lower["score"]
        upper_scores = df_upper["score"]

        lower_scores = lower_scores.where(lower_scores >= 1e-10, 0)
        
        sharpness = upper_scores - lower_scores
        qhat = sharpness / 2.0
        coverage_mask = (labels >= lower_scores) & (labels <= upper_scores)
        num_total = len(labels)
        coverage_combined = coverage_mask.sum() / num_total if num_total > 0 else float('nan')
        sharpness_combined = sharpness.mean() if num_total > 0 else float('nan')
        qhat_combined = qhat.mean() if num_total > 0 else float('nan')

        pos_mask = labels > 0
        num_pos = pos_mask.sum()
        if num_pos > 0:
            coverage_pos = (coverage_mask[pos_mask]).sum() / num_pos
            sharpness_pos = sharpness[pos_mask].mean()
            qhat_pos = qhat[pos_mask].mean()
        else:
            coverage_pos = float('nan')
            sharpness_pos = float('nan')
            qhat_pos = float('nan')

        neg_mask = labels == 0
        num_neg = neg_mask.sum()
        if num_neg > 0:
            coverage_neg = (coverage_mask[neg_mask]).sum() / num_neg
            sharpness_neg = sharpness[neg_mask].mean()
            qhat_neg = qhat[neg_mask].mean()
        else:
            coverage_neg = float('nan')
            sharpness_neg = float('nan')
            qhat_neg = float('nan')
        
        new_row = {
            'coverage_pos': coverage_pos,
            'sharpness_pos': sharpness_pos,
            'qhat_pos': qhat_pos,
            'coverage_neg': coverage_neg,
            'sharpness_neg': sharpness_neg,
            'qhat_neg': qhat_neg,
            'coverage_combined': coverage_combined,
            'sharpness_combined': sharpness_combined,
            'qhat_combined': qhat_combined
        }

        if dataset not in results:
            results[dataset] = []
        results[dataset].append(new_row)

for dataset, rows in results.items():
    df_result = pd.DataFrame(rows)
    output_filename = f"quan_{dataset}.csv"
    df_result.to_csv(output_filename, index=False)
    print(f"{output_filename}")