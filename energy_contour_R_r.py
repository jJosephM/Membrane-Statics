import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Use LaTeX to enable custom fonts
plt.rcParams["text.usetex"] = True

# --- File path ---
csv_path = "R_vs_r(1).csv"   # adjust if needed

# --- Load & coerce numeric ---
df = pd.read_csv(csv_path)
df_num = df.copy()
for c in df_num.columns:
    df_num[c] = pd.to_numeric(df_num[c], errors='coerce')

# --- Identify columns ---
# First two numeric columns for grid:
num_cols = [c for c in df_num.columns if np.issubdtype(df_num[c].dtype, np.number)]
if len(num_cols) < 3:
    raise ValueError("Need at least three numeric columns (R, r, and E).")

xcol, ycol = num_cols[0], num_cols[1]

# Use 'E' for the value column if present, otherwise the 3rd numeric:
zcol = 'beta' if 'beta' in df_num.columns else num_cols[7]

# --- Build gridded table (handles duplicates by averaging) ---
grid = df_num.pivot_table(index=ycol, columns=xcol, values=zcol, aggfunc='mean')

# Drop rows/cols that are entirely NaN (just in case)
grid = grid.dropna(axis=0, how='all').dropna(axis=1, how='all')

# --- Create mesh ---
X, Y = np.meshgrid(grid.columns.values, grid.index.values)
Z = grid.values

# --- Plot ---
plt.figure(figsize=(3.7,3))
cs = plt.contourf(X, Y, Z, levels=50)
cbar = plt.colorbar(cs, label=r'$J\Delta_0$')

# ax = plt.gca()             # get current axis
# ax.yaxis.set_visible(False)   # hides both label and ticks
# ax.yaxis.tick_right()      # move ticks to the right
# ax.yaxis.set_label_position("right")  # move label to the right

plt.xlabel(r'$R/\Delta_0$',size=13)
plt.ylabel(r'$r/\Delta_0$',size=13)
plt.title(rf'$\Pi^\Phi$ - Term',size=13)
plt.tight_layout()
plt.savefig(f"beta_contour.png", dpi=300)
# plt.show()
