"""
AI4I 2020 Predictive Maintenance - Exploratory Data Analysis (EDA)
==================================================================
Generates high-resolution publication-quality visualization charts
for feature distributions, outlier analysis, failure modes, correlation,
and physics-based feature engineering relationships.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set high-res styling and typography
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.2,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "axes.labelcolor": "#1E293B",
    "figure.titlesize": 14,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

PRIMARY_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
TARGET = "Machine failure"

PALETTE = {
    "normal": "#10B981",    # Emerald Green for Normal
    "failure": "#EF4444",   # Crimson Red for Failure
    "primary": "#3B82F6",   # Vibrant Blue
    "accent": "#8B5CF6",    # Purple
    "dark": "#1E293B"       # Slate
}

def get_data_and_output_paths():
    """Resolves raw data path and chart output directory."""
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "raw" / "ai4i2020.csv"
    if not data_path.exists():
        data_path = base_dir.parent / "ai4i2020.csv"
    
    chart_dir = base_dir / "outputs" / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    return data_path, chart_dir

def plot_target_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot 1: Machine failure class distribution with counts and percentages."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    counts = df[TARGET].value_counts().sort_index()
    total = len(df)
    labels = ["Normal Operational (0)", "Machine Failure (1)"]
    colors = [PALETTE["normal"], PALETTE["failure"]]
    
    bars = ax.bar(labels, counts.values, color=colors, width=0.45, edgecolor="#0F172A", linewidth=1.2)
    ax.set_title("Machine Failure Class Distribution (Target Imbalance)", pad=15, fontweight="bold")
    ax.set_ylabel("Number of Records", fontweight="bold")
    ax.set_ylim(0, total * 1.1)
    ax.grid(axis="y")
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(f"{height:,} records\n({pct:.2f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontweight="bold", fontsize=10, color="#1E293B")
        
    ratio = counts[0] / counts[1]
    ax.text(0.98, 0.85, f"Imbalance Ratio: {ratio:.1f} : 1\nNormal: {counts[0]:,} ({counts[0]/total*100:.1f}%)\nFailure: {counts[1]:,} ({counts[1]/total*100:.1f}%)",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9),
            fontsize=10)
    
    plt.tight_layout()
    save_path = output_dir / "01_machine_failure_distribution.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved: {save_path.name}")

def plot_single_distribution(df: pd.DataFrame, col: str, unit: str, output_path: Path, chart_num: str):
    """Plots histogram + KDE for a single continuous sensor feature."""
    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                          gridspec_kw={"height_ratios": [0.2, 0.8]}, dpi=300)
    
    # Top boxplot
    sns.boxplot(x=df[col], ax=ax_box, color="#93C5FD", fliersize=3,
                boxprops=dict(edgecolor="#1E3A8A"), whiskerprops=dict(color="#1E3A8A"))
    ax_box.set(xlabel="")
    ax_box.set_title(f"{col} - Distribution & Outlier Overview", fontweight="bold", pad=12)
    ax_box.grid(axis="x")
    
    # Bottom histogram + KDE
    sns.histplot(df[col], kde=True, ax=ax_hist, color=PALETTE["primary"],
                 edgecolor="#1E3A8A", alpha=0.65, bins=40, line_kws={"linewidth": 2.2, "color": "#1E3A8A"})
    
    mean_val = df[col].mean()
    median_val = df[col].median()
    std_val = df[col].std()
    
    ax_hist.axvline(mean_val, color="#DC2626", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:.2f} {unit}")
    ax_hist.axvline(median_val, color="#16A34A", linestyle="-.", linewidth=1.8, label=f"Median: {median_val:.2f} {unit}")
    
    ax_hist.set_xlabel(f"{col}", fontweight="bold")
    ax_hist.set_ylabel("Frequency / Count", fontweight="bold")
    ax_hist.legend(loc="upper right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    ax_hist.grid(axis="both")
    
    # Statistical annotations
    q25 = df[col].quantile(0.25)
    q75 = df[col].quantile(0.75)
    iqr = q75 - q25
    stats_text = f"Std Dev: {std_val:.2f} {unit}\nIQR: {iqr:.2f} {unit}\nMin: {df[col].min():.1f} | Max: {df[col].max():.1f}"
    ax_hist.text(0.03, 0.93, stats_text, transform=ax_hist.transAxes, va="top",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9),
                 fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved: {output_path.name}")

def plot_sensor_boxplots(df: pd.DataFrame, output_dir: Path):
    """Plot 7: Standardized box plots for all five primary features to examine outliers."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 5), dpi=300)
    
    for i, col in enumerate(PRIMARY_FEATURES):
        ax = axes[i]
        sns.boxplot(y=df[col], ax=ax, color="#BAE6FD", flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#EF4444", "markeredgecolor": "none", "alpha": 0.6})
        ax.set_title(col.split("[")[0].strip(), fontweight="bold", fontsize=11)
        ax.set_ylabel(col, fontweight="bold", fontsize=10)
        ax.grid(axis="y")
        
        # Calculate outlier count via IQR
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_pct = (len(outliers) / len(df)) * 100
        
        ax.set_xlabel(f"IQR Outliers:\n{len(outliers)} ({outlier_pct:.1f}%)", fontsize=9, color="#B91C1C" if len(outliers) > 0 else "#15803D", fontweight="bold")
        
    plt.suptitle("Outlier Analysis via Interquartile Range (IQR) for 5 Sensor Features", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    save_path = output_dir / "07_sensor_features_boxplots.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved: {save_path.name}")

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    """Plot 8: Correlation heatmap for primary sensor features and target variable."""
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    cols_to_corr = PRIMARY_FEATURES + [TARGET]
    corr_matrix = df[cols_to_corr].corr(method="pearson")
    
    # Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, annot=True,
                fmt=".3f", linewidths=1.0, linecolor="white", cbar_kws={"label": "Pearson Correlation Coefficient", "shrink": 0.8},
                ax=ax)
    
    ax.set_title("Pearson Correlation Matrix: Sensor Telemetry vs Machine Failure", pad=15, fontweight="bold")
    plt.tight_layout()
    save_path = output_dir / "08_correlation_heatmap.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved: {save_path.name}")

def plot_sensor_comparison_by_failure(df: pd.DataFrame, output_dir: Path):
    """Plot 9: Comparison of sensor features distributions split by Machine failure status."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    axes = axes.flatten()
    
    for i, col in enumerate(PRIMARY_FEATURES):
        ax = axes[i]
        sns.kdeplot(data=df[df[TARGET] == 0], x=col, ax=ax, label="Normal (0)",
                    color=PALETTE["normal"], fill=True, alpha=0.35, linewidth=2)
        sns.kdeplot(data=df[df[TARGET] == 1], x=col, ax=ax, label="Failure (1)",
                    color=PALETTE["failure"], fill=True, alpha=0.45, linewidth=2)
        
        ax.set_title(f"{col}", fontweight="bold", fontsize=11)
        ax.set_xlabel(col, fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.grid(True)
        ax.legend(loc="upper right", frameon=True)
        
    # Sixth subplot: torque vs rotational speed scatter highlighting failures
    ax_scatter = axes[5]
    sns.scatterplot(data=df[df[TARGET] == 0], x="Rotational speed [rpm]", y="Torque [Nm]",
                    ax=ax_scatter, color=PALETTE["normal"], alpha=0.25, s=15, label="Normal (0)")
    sns.scatterplot(data=df[df[TARGET] == 1], x="Rotational speed [rpm]", y="Torque [Nm]",
                    ax=ax_scatter, color=PALETTE["failure"], alpha=0.85, s=30, edgecolor="black", label="Failure (1)")
    ax_scatter.set_title("Torque vs Speed (Failure Clusters)", fontweight="bold", fontsize=11)
    ax_scatter.grid(True)
    ax_scatter.legend(loc="upper right", frameon=True)
    
    plt.suptitle("Sensor Feature Telemetry Comparison: Normal vs Machine Failure", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    save_path = output_dir / "09_sensor_comparison_by_failure.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved: {save_path.name}")

def plot_pairplot(df: pd.DataFrame, output_dir: Path):
    """Plot 10: Pair plot of primary features colored by failure class."""
    cols = PRIMARY_FEATURES + [TARGET]
    
    # Subsample normal cases slightly for rendering performance if needed, or render cleanly
    # 10,000 points is very fast in seaborn with s=12
    g = sns.pairplot(
        df[cols],
        hue=TARGET,
        palette={0: PALETTE["normal"], 1: PALETTE["failure"]},
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 15},
        diag_kws={"fill": True, "alpha": 0.4},
        corner=True,
        height=2.2
    )
    g.fig.suptitle("Pairwise Relationships Among Sensor Features Colored by Machine Failure", y=1.02, fontweight="bold", fontsize=14)
    save_path = output_dir / "10_pairplot_feature_relationships.png"
    g.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved: {save_path.name}")

def plot_engineered_features(df: pd.DataFrame, output_dir: Path):
    """Plot 11: Physics-based engineered features analysis."""
    df_eng = df.copy()
    df_eng["delta_temperature"] = df_eng["Process temperature [K]"] - df_eng["Air temperature [K]"]
    df_eng["mechanical_power"] = df_eng["Torque [Nm]"] * df_eng["Rotational speed [rpm]"] * 2 * np.pi / 60
    df_eng["overstrain_index"] = df_eng["Torque [Nm]"] * df_eng["Tool wear [min]"]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    eng_vars = [
        ("delta_temperature", "Temperature Difference ΔT [K]", "ΔT < 8.6 K triggers HDF threshold"),
        ("mechanical_power", "Mechanical Power [W]", "Safe: 3,500 W - 9,000 W (PWF limits)"),
        ("overstrain_index", "Overstrain Index [Nm·min]", "High torque + tool wear triggers OSF")
    ]
    
    for i, (col, title, note) in enumerate(eng_vars):
        ax = axes[i]
        sns.kdeplot(data=df_eng[df_eng[TARGET] == 0], x=col, ax=ax, label="Normal (0)",
                    color=PALETTE["normal"], fill=True, alpha=0.35, linewidth=2)
        sns.kdeplot(data=df_eng[df_eng[TARGET] == 1], x=col, ax=ax, label="Failure (1)",
                    color=PALETTE["failure"], fill=True, alpha=0.45, linewidth=2)
        
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel(col, fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.grid(True)
        ax.legend(loc="upper right", frameon=True)
        ax.text(0.5, -0.22, note, transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#475569")
        
    plt.suptitle("Physics-Based Engineered Features vs Machine Failure (Experimental Study)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = output_dir / "11_engineered_features_analysis.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved: {save_path.name}")

def plot_failure_modes_breakdown(df: pd.DataFrame, output_dir: Path):
    """Plot 12: Breakdown of specific failure modes."""
    failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    counts = [int(df[col].sum()) for col in failure_cols]
    descriptions = [
        "TWF: Tool Wear (46)",
        "HDF: Heat Dissipation (115)",
        "PWF: Power Failure (95)",
        "OSF: Overstrain (98)",
        "RNF: Random Failure (19)"
    ]
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    bars = ax.bar(failure_cols, counts, color=["#F59E0B", "#EF4444", "#3B82F6", "#8B5CF6", "#64748B"],
                  width=0.45, edgecolor="#0F172A", linewidth=1.1)
    
    ax.set_title("Distribution of Specific Failure Modes (Sub-Labels)", pad=15, fontweight="bold")
    ax.set_ylabel("Failure Occurrence Count", fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.25)
    ax.grid(axis="y")
    
    for bar, count in zip(bars, counts):
        pct_of_total_failures = (count / df[TARGET].sum()) * 100
        ax.annotate(f"{count}\n({pct_of_total_failures:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, count),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=10)
        
    ax.text(0.98, 0.88,
            "Note: Total Machine Failures = 339.\n"
            "Some records experience multiple failure modes simultaneously.\n"
            "Excluded from features to prevent direct target leakage.",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7", edgecolor="#F59E0B", alpha=0.9),
            fontsize=9)
    
    plt.tight_layout()
    save_path = output_dir / "12_failure_modes_breakdown.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved: {save_path.name}")

def run_all_eda():
    data_path, output_dir = get_data_and_output_paths()
    print(f"[EDA] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"[EDA] Generating visualizations in: {output_dir}")
    
    # 1. Target distribution
    plot_target_distribution(df, output_dir)
    
    # 2 - 6. Individual feature distributions
    feature_meta = [
        ("Air temperature [K]", "K", "02_air_temperature_distribution.png", "02"),
        ("Process temperature [K]", "K", "03_process_temperature_distribution.png", "03"),
        ("Rotational speed [rpm]", "rpm", "04_rotational_speed_distribution.png", "04"),
        ("Torque [Nm]", "Nm", "05_torque_distribution.png", "05"),
        ("Tool wear [min]", "min", "06_tool_wear_distribution.png", "06"),
    ]
    for col, unit, filename, num in feature_meta:
        plot_single_distribution(df, col, unit, output_dir / filename, num)
        
    # 7. Boxplots
    plot_sensor_boxplots(df, output_dir)
    
    # 8. Correlation heatmap
    plot_correlation_heatmap(df, output_dir)
    
    # 9. Comparison by failure
    plot_sensor_comparison_by_failure(df, output_dir)
    
    # 10. Pairplot
    plot_pairplot(df, output_dir)
    
    # 11. Engineered features
    plot_engineered_features(df, output_dir)
    
    # 12. Failure modes breakdown
    plot_failure_modes_breakdown(df, output_dir)
    
    print("[EDA] All 12 visualization charts successfully created!")

if __name__ == "__main__":
    run_all_eda()
