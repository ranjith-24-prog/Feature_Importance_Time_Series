import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- LOAD & PREPARE DATA ----------
df = pd.read_csv("output/FI_Comparison_Results/Comparison_Results_WinIT.csv")  # Change to your target file
df["Dataset Name"] = df["Dataset"].apply(lambda x: x.split("/")[-1].replace(".csv", ""))
datasets = ["DMC2_AL_CP1", "DMC2_AL_CP2", "DMC2_S_CP1", "DMC2_S_CP2"]
df = df[df["Dataset Name"].isin(datasets)]

# ---------- TECHNIQUE ORDER & COLOR PALETTE ----------
techniques = df["Technique"].drop_duplicates().tolist()  # preserve original order
palette = sns.color_palette("tab20", len(techniques))
color_dict = dict(zip(techniques, palette))

# ---------- PLOT SETUP ----------
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# ---------- LEFT: GROUPED TEST LOSS ----------
sns.barplot(
    data=df,
    x="Dataset Name",
    y="Test Loss",
    hue="Technique",
    ax=axes[0],
    palette=color_dict
)
axes[0].set_title("Test Loss by Technique per Dataset")
axes[0].set_xlabel("Dataset")
axes[0].set_ylabel("Test Loss")

# Add value labels to test loss bars
for container in axes[0].containers:
    axes[0].bar_label(container, fmt="%.4f", label_type="edge", fontsize=9)

# ---------- RIGHT: GROUPED EXECUTION TIME ----------
sns.barplot(
    data=df,
    x="Dataset Name",
    y="Execution Time (Seconds)",
    hue="Technique",
    ax=axes[1],
    palette=color_dict
)
axes[1].set_title("Execution Time by Technique per Dataset")
axes[1].set_xlabel("Dataset")
axes[1].set_ylabel("Execution Time (Seconds)")

# Add value labels to execution bars
for container in axes[1].containers:
    axes[1].bar_label(container, fmt="%d", label_type="edge", fontsize=9)

# ---------- SINGLE LEGEND BELOW ----------
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="Technique",
    loc="lower center",
    ncol=3,
    fontsize=10,
    frameon=False
)

# Remove individual legends
if axes[1].get_legend() is not None:
    axes[1].get_legend().remove()
if axes[0].get_legend() is not None:
    axes[0].get_legend().remove()

# ---------- FINAL TOUCHES ----------
plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave space for legend
plt.savefig("output/FI_Comparison_Plots/comparison_plot_WinIT.png", dpi=300, bbox_inches='tight')
plt.show()