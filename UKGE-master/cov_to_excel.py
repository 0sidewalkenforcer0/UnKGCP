import pandas as pd
import os
import math

def custom_round(x):
    x_rounded = round(x, 2)
    if abs((x * 100) % 1 - 0.5) < 1e-6:
        x_rounded = math.ceil(x * 100) / 100
    return f"{x_rounded:.2f}"

datasets = ['cn15k', 'nl27k', 'ppi5k']
models = ['logi', 'rect']
subset_sizes = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]

for file_path in [
    f"UKGE_{dataset}_{model}_randomaccum_subset_{size}.csv"
    for dataset in datasets
    for model in models
    for size in subset_sizes
]:
    df = pd.read_csv(file_path)

    stats = df.agg(['mean', 'std']).T.reset_index()
    stats.columns = ["Metric", "Mean", "Std Dev"]

    stats["Formatted"] = stats.apply(lambda row: f"{custom_round(row['Mean'])} ({row['Std Dev']:.3f})", axis=1)

    final_col = stats[["Metric", "Formatted"]]

    final_col = final_col.set_index("Metric")

    output_file = file_path.replace(".csv", "_transposed.xlsx")
    final_col.to_excel(output_file)

    if os.name == 'nt':
        os.system(f'start excel "{output_file}"')

print("Done")