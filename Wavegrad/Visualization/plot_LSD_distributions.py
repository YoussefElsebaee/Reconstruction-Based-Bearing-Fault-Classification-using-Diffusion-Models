import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import json
import os


# ==========================================================
# Configuration
# ==========================================================

RESULTS_PATH = Path(
    r"C:\Users\yosef\OneDrive - GIU AS - German International University of Applied Sciences\Desktop\metrics (LSD)\8dB noise\classification_results_LSD.xlsx")

LABEL_COL = "true_class"
SAMPLE_COL = "sample"

MODEL_LOSS_COLUMNS = {
    "Normal model": "LSD_normal",
    "Ball model": "LSD_ball",
    "Inner model": "LSD_inner",
    "Outer model": "LSD_outer",
}

CLASS_NAMES = {
    0: "Normal",
    1: "Ball Fault",
    2: "Inner-Race Fault",
    3: "Outer-Race Fault",
}

MODEL_COLORS = {
    "Normal model": "#4F4F4F",  # dark grey
    "Ball model": "#FF9800",    # orange
    "Inner model": "#1976D2",   # blue
    "Outer model": "#C62828",   # red
}

MODEL_FILL_COLORS = {
    "Normal model": "#B0B0B0",  # light grey
    "Ball model": "#FFB74D",    # light orange
    "Inner model": "#64B5F6",   # light blue
    "Outer model": "#E57373",   # light red
}

CLASS_COLOR = {
    0: MODEL_COLORS["Normal model"],
    1: MODEL_COLORS["Ball model"],
    2: MODEL_COLORS["Inner model"],
    3: MODEL_COLORS["Outer model"]
}


FIGSIZE_FAULTED = (14, 9)
FIGSIZE_NORMAL = (14, 4)

ALPHA_FILL = 0.30
LINEWIDTH = 1.8

# ==========================================================
# Load data
# ==========================================================

if not RESULTS_PATH.exists():
    raise FileNotFoundError(f"Results file not found: {RESULTS_PATH}")

df = pd.read_excel(RESULTS_PATH)

required_cols = [LABEL_COL, SAMPLE_COL] + list(MODEL_LOSS_COLUMNS.values())
missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in Excel file: {missing}")

# ==========================================================
# Parse sample column (fault type, fault size, load)
# ==========================================================

def parse_sample(sample_name):
    """
    Robust parser for CWRU-style filenames.

    Examples:
    - Normal_0_chunk003.wav
    - B007_0_chunk003.wav
    - IR014_2_chunk140.wav
    - OR007@6_0_chunk003.wav

    Returns:
    fault_type, fault_size (None for Normal), load
    """

    # -------------------------
    # Normal samples
    # -------------------------
    if sample_name.startswith("Normal"):
        match = re.search(r"Normal_(\d)_chunk", sample_name)
        if match is None:
            raise ValueError(f"Unrecognized Normal format: {sample_name}")
        return "Normal", None, int(match.group(1))

    # -------------------------
    # Outer race (special @6)
    # -------------------------
    if sample_name.startswith("OR"):
        match = re.search(r"OR(\d{3})@6_(\d)_chunk", sample_name)
        if match is None:
            raise ValueError(f"Unrecognized Outer Race format: {sample_name}")

        fault_size = float(match.group(1)) / 1000.0
        load = int(match.group(2))
        return "Outer", fault_size, load

    # -------------------------
    # Ball & Inner race
    # -------------------------
    match = re.search(r"([A-Za-z]+)(\d{3})_(\d)_chunk", sample_name)
    if match is None:
        raise ValueError(f"Unrecognized faulted sample format: {sample_name}")

    fault_code = match.group(1).upper()
    fault_size = float(match.group(2)) / 1000.0
    load = int(match.group(3))

    fault_map = {
        "B": "Ball",
        "IR": "Inner",
    }

    if fault_code not in fault_map:
        raise ValueError(f"Unknown fault code '{fault_code}' in {sample_name}")

    return fault_map[fault_code], fault_size, load


df[["fault_type", "fault_size", "load"]] = (
    df[SAMPLE_COL].apply(parse_sample).apply(pd.Series)
)

# ==========================================================
# Plotting function (Strategy 1: Small Multiples)
# ==========================================================

def plot_class_distributions(df, target_class, save_path=r"C:\Users\yosef\OneDrive - GIU AS - German International University of Applied Sciences\Desktop\metrics (LSD)\8dB noise\KDE_distributions"):
    class_df = df[df[LABEL_COL] == target_class]
    is_normal = target_class == 0

    loads = sorted(class_df["load"].unique())
    fault_sizes = [None] if is_normal else sorted(class_df["fault_size"].unique())

    n_rows = len(fault_sizes)
    n_cols = len(loads)

    # Initialize means dictionary
    all_means = {f"class_{target_class}": {}}

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=FIGSIZE_NORMAL if is_normal else FIGSIZE_FAULTED,
        sharex=True, sharey=True
    )

    if n_rows == 1:
        axes = np.array([axes])

    for i, fault in enumerate(fault_sizes):
        for j, load in enumerate(loads):
            ax = axes[i, j]
            subset = class_df[class_df["load"] == load]
            if not is_normal:
                subset = subset[subset["fault_size"] == fault]

            for model_name, loss_col in MODEL_LOSS_COLUMNS.items():
                data = subset[loss_col].dropna()
                if len(data) < 5:
                    continue

                fill_color = MODEL_FILL_COLORS[model_name]
                line_color = MODEL_COLORS[model_name]
                mean_val = data.mean()

                # Save mean in dictionary
                cls_key = all_means[f"class_{target_class}"]
                fault_key = str(fault) if fault is not None else "None"
                if model_name not in cls_key:
                    cls_key[model_name] = {}
                if fault_key not in cls_key[model_name]:
                    cls_key[model_name][fault_key] = {}
                cls_key[model_name][fault_key][str(load)] = float(mean_val)

                # KDE plot
                sns.kdeplot(
                    data,
                    ax=ax,
                    fill=True,
                    alpha=0.4,
                    linewidth=1.8,
                    color=line_color,
                    bw_adjust=0.6
                )

            # Axes aesthetics
            ax.grid(True, linestyle='-', linewidth=0.8, color='0.7', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.tick_params(direction='out', length=6, width=1.2)
            ax.set_xlabel('')
            ax.set_ylabel('')

            if i == 0:
                ax.set_title(f"{load} HP", fontsize=10, fontweight="bold")
            if j == 0 and not is_normal:
                ax.set_ylabel(f'{fault}"', fontsize=10, fontweight="bold")
    # Reserve space for top and bottom
    plt.subplots_adjust(
        top=0.88,    # space for top title and class name
        bottom=0.12, # space for xlabel
        left=0.12,   # space for ylabel
        right=0.95
    )

    # Class name, colored
    #fig.text(0.5, 0.94, CLASS_NAMES[target_class], color=CLASS_COLOR[target_class], fontsize=16, fontweight='bold', ha='center')

    fig.supxlabel("LSD", fontsize=12)
    fig.supylabel("Density", fontsize=12, y=0.6)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=MODEL_FILL_COLORS[m], edgecolor=MODEL_COLORS[m], label=m)
        for m in MODEL_COLORS.keys()
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper right',
        frameon=True,
        framealpha=0.9,
        shadow=True,
        fontsize=10,  # slightly smaller
        title="Models",
        title_fontsize=11
    )

    plt.tight_layout()
    # Save plot
    plot_filename = os.path.join(save_path, f"{CLASS_NAMES[target_class].replace(' ', '_')}.png")
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot for class '{CLASS_NAMES[target_class]}' at: {plot_filename}")
    plt.show()

    # Save means to JSON
    save_means_path = os.path.join(save_path, "means.json")
    with open(save_means_path, 'w') as f:
        json.dump(all_means, f, indent=4)




# ==========================================================
# Main
# ==========================================================

def main():
    for cls in CLASS_NAMES.keys():
        plot_class_distributions(df, cls)

if __name__ == "__main__":
    main()
