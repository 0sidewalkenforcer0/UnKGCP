import pandas as pd
import matplotlib.pyplot as plt

file_path = "UKGE_ppi5k_rect.csv"
df = pd.read_csv(file_path)
stats = df.agg(['mean'])
stats = stats.T.reset_index()
stats.columns = ["Metric", "Mean"]
stats = stats.round(6)

fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=stats.values, colLabels=stats.columns, cellLoc='center', loc='center')

plt.show()