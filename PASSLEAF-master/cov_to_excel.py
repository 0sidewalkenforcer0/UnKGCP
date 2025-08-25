import pandas as pd
import os
import math

def custom_round(x):
    x_rounded = round(x, 2)
    if abs((x * 100) % 1 - 0.5) < 1e-6:
        x_rounded = math.ceil(x * 100) / 100
    return f"{x_rounded:.2f}"

for file_path in [
    "PASSLEAF_cn15k_randomaccum_subset_10.csv",
    "PASSLEAF_cn15k_randomaccum_subset_20.csv",
    "PASSLEAF_cn15k_randomaccum_subset_40.csv",
    "PASSLEAF_cn15k_randomaccum_subset_80.csv",
    "PASSLEAF_cn15k_randomaccum_subset_160.csv",
    "PASSLEAF_cn15k_randomaccum_subset_320.csv",
    "PASSLEAF_cn15k_randomaccum_subset_640.csv",
    "PASSLEAF_cn15k_randomaccum_subset_1280.csv",
    "PASSLEAF_cn15k_randomaccum_subset_2560.csv",
    "PASSLEAF_cn15k_randomaccum_subset_5120.csv",
    "PASSLEAF_cn15k_randomaccum_subset_10240.csv",
    
    "PASSLEAF_nl27k_randomaccum_subset_10.csv",
    "PASSLEAF_nl27k_randomaccum_subset_20.csv",
    "PASSLEAF_nl27k_randomaccum_subset_40.csv",
    "PASSLEAF_nl27k_randomaccum_subset_80.csv",
    "PASSLEAF_nl27k_randomaccum_subset_160.csv",
    "PASSLEAF_nl27k_randomaccum_subset_320.csv",
    "PASSLEAF_nl27k_randomaccum_subset_640.csv",
    "PASSLEAF_nl27k_randomaccum_subset_1280.csv",
    "PASSLEAF_nl27k_randomaccum_subset_2560.csv",
    "PASSLEAF_nl27k_randomaccum_subset_5120.csv",
    "PASSLEAF_nl27k_randomaccum_subset_10240.csv",

    "PASSLEAF_ppi5k_randomaccum_subset_10.csv",
    "PASSLEAF_ppi5k_randomaccum_subset_20.csv",
    "PASSLEAF_ppi5k_randomaccum_subset_40.csv",
    "PASSLEAF_ppi5k_randomaccum_subset_80.csv",
    "PASSLEAF_ppi5k_randomaccum_subset_160.csv",
    "PASSLEAF_ppi5k_randomaccum_subset_320.csv",
    "PASSLEAF_ppi5k_randomaccum_subset_640.csv",
    "PASSLEAF_ppi5k_randomaccum_subset_1280.csv",
    "PASSLEAF_ppi5k_randomaccum_subset_2560.csv",
    "PASSLEAF_ppi5k_randomaccum_subset_5120.csv",
    "PASSLEAF_ppi5k_randomaccum_subset_10240.csv"
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