#!/usr/bin/env python3
"""
Visualization script for Smart Grid Real-Time Load Monitoring and Peak Analysis.
Generates publication-quality charts for the Hybrid MPI + OpenMP implementation.
"""

import os
import sys
import argparse
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200
})

def plot_hourly_profile(hourly_csv, out_dir):
    if not os.path.exists(hourly_csv):
        print(f"[-] Hourly stats file not found: {hourly_csv}")
        return
    df = pd.read_csv(hourly_csv)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    hours = df["Hour"]
    mean_p = df["Mean_Power_kW"]
    std_p = df["StdDev_kW"]
    max_p = df["Max_Power_kW"]
    min_p = df["Min_Power_kW"]

    # Mean line and stddev envelope
    ax1.plot(hours, mean_p, color="#1f77b4", marker="o", linewidth=2.5, label="Hourly Mean Load (kW)")
    ax1.fill_between(hours, np.maximum(0, mean_p - std_p), mean_p + std_p, color="#1f77b4", alpha=0.2, label=r"Mean $\pm 1\sigma$ Band")

    # Max power envelope
    ax1.plot(hours, max_p, color="#d62728", linestyle="--", marker="s", markersize=5, label="Hourly Maximum Spike (kW)")
    ax1.plot(hours, min_p, color="#2ca02c", linestyle=":", marker="^", markersize=4, label="Hourly Minimum (kW)")

    # Secondary axis for total energy (kWh) and renewable generation
    ax2 = ax1.twinx()
    bar_width = 0.35
    ax2.bar(hours - bar_width/2, df["Total_Energy_kWh"], color="#ff7f0e", alpha=0.35, width=bar_width, label="Grid Load Energy (kWh)")
    if "Renewable_kWh" in df:
        ax2.bar(hours + bar_width/2, df["Renewable_kWh"], color="#2ca02c", alpha=0.35, width=bar_width, label="Renewable Energy (kWh)")
    ax2.set_ylabel("Total Energy (kWh)", color="#d95f02")
    ax2.tick_params(axis="y", labelcolor="#d95f02")
    ax2.grid(False)

    ax1.set_xlabel("Diurnal Hour of Day (00:00 - 23:00)")
    ax1.set_ylabel("Active Power (kW)")
    ax1.set_title("Hybrid MPI+OpenMP: Diurnal Load Profile, Extremes, & Renewable Generation")
    ax1.set_xticks(range(0, 24))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig1_hourly_power_profile.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")

def plot_daily_consumption(daily_csv, out_dir):
    if not os.path.exists(daily_csv):
        print(f"[-] Daily stats file not found: {daily_csv}")
        return
    df = pd.read_csv(daily_csv)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

    dates = df["Date"]
    x_indices = np.arange(len(dates))

    # Top: Daily Mean & Max Power
    ax1.plot(x_indices, df["Mean_Power_kW"], marker="o", color="#2b5c8f", linewidth=2, label="Daily Mean Load (kW)")
    ax1.plot(x_indices, df["Max_Power_kW"], marker="^", color="#d62728", linestyle="--", label="Daily Maximum Peak (kW)")
    ax1.set_ylabel("Power (kW)")
    ax1.set_title("Hybrid MPI+OpenMP: Daily Grid Demand Trends & Maximum Load Spikes")
    ax1.legend(loc="upper right", frameon=True)

    # Bottom: Daily Energy & Renewable Generation
    ax2.bar(x_indices, df["Total_Energy_kWh"], color="#4e79a7", alpha=0.7, label="Grid Energy Consumed (kWh)")
    if "Renewable_kWh" in df:
        ax2.plot(x_indices, df["Renewable_kWh"], color="#59a14f", marker="s", linewidth=1.8, label="Solar + Wind Energy (kWh)")
    ax2.set_ylabel("Energy (kWh)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", frameon=True)

    step = max(1, len(dates) // 12)
    ax2.set_xticks(x_indices[::step])
    ax2.set_xticklabels(dates[::step], rotation=35, ha="right")

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig2_daily_consumption.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")

def plot_anomalies_distribution(anomalies_csv, summary_csv, out_dir):
    if not os.path.exists(anomalies_csv):
        print(f"[-] Anomalies file not found: {anomalies_csv}")
        return
    df_anom = pd.read_csv(anomalies_csv)

    threshold = 0.0
    mean_power = 0.0
    if os.path.exists(summary_csv):
        s_df = pd.read_csv(summary_csv)
        s_dict = dict(zip(s_df["Metric"], s_df["Value"]))
        threshold = float(s_dict.get("Anomaly_Threshold", 0.0))
        mean_power = float(s_dict.get("Mean_Active_Power", 0.0))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    if len(df_anom) > 0:
        df_plot = df_anom.sample(n=min(len(df_anom), 2000), random_state=42).sort_values("Timestamp") if len(df_anom) > 2000 else df_anom
        scatter = ax.scatter(
            range(len(df_plot)),
            df_plot["Active_Power_kW"],
            c=df_plot["Z_Score"],
            cmap="plasma",
            s=35,
            edgecolors="black",
            linewidth=0.4,
            alpha=0.85,
            label="Detected Grid Anomalies / Overloads"
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Z-Score ($z = (x - \\mu)/\\sigma$)")
    else:
        ax.text(0.5, 0.5, "No anomalies flagged exceeding threshold", ha="center", va="center", transform=ax.transAxes)

    if threshold > 0:
        ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.8, label=f"Anomaly Threshold ({threshold:.2f} kW)")
    if mean_power > 0:
        ax.axhline(mean_power, color="#1f77b4", linestyle=":", linewidth=1.5, label=f"Mean Active Power ({mean_power:.2f} kW)")

    ax.set_xlabel("Flagged Incident Index")
    ax.set_ylabel("Power Consumption (kW)")
    ax.set_title("Hybrid MPI+OpenMP: Grid Outliers, Overloads & Extreme Load Anomalies")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig3_anomalies_distribution.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")

def plot_benchmarks(benchmark_csv, out_dir):
    if not os.path.exists(benchmark_csv):
        print(f"[-] Benchmark CSV not found: {benchmark_csv}. Skipping scaling plots.")
        return

    df = pd.read_csv(benchmark_csv)
    if len(df) == 0:
        return

    # Figure 4: Stacked Execution Time Breakdown
    fig, ax = plt.subplots(figsize=(11, 5.5))
    configs = df["Config_Name"]
    x = np.arange(len(configs))
    width = 0.5

    t_load = df["T_Load_s"] if "T_Load_s" in df else np.zeros(len(df))
    t_dist = df["T_Dist_s"] if "T_Dist_s" in df else np.zeros(len(df))
    t_comp = df["T_Comp_s"]
    t_comm = df["T_Comm_s"] if "T_Comm_s" in df else np.zeros(len(df))

    ax.bar(x, t_load, width, label="Data Ingestion & Parsing", color="#4e79a7")
    ax.bar(x, t_dist, width, bottom=t_load, label="MPI Data Scatterv", color="#f28e2b")
    ax.bar(x, t_comp, width, bottom=t_load + t_dist, label="OpenMP Threads Computation", color="#59a14f")
    ax.bar(x, t_comm, width, bottom=t_load + t_dist + t_comp, label="MPI Collective Reductions", color="#e15759")

    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Hybrid MPI + OpenMP: Execution Time Breakdown Across Configurations")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=16, ha="right")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig4_execution_time_breakdown.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")

    # Figure 5: Speedup & Parallel Efficiency Scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    units = df["Total_Units"]
    speedup_comp = df["Speedup_Comp"] if "Speedup_Comp" in df else df.get("Speedup", units * 0 + 1.0)
    speedup_e2e = df["Speedup_E2E"] if "Speedup_E2E" in df else speedup_comp
    efficiency_comp = (df["Efficiency_Comp"] * 100.0) if "Efficiency_Comp" in df else (df.get("Efficiency", units * 0 + 1.0) * 100.0)

    # Speedup
    ax1.plot(units, speedup_comp, marker="o", linewidth=2.4, color="#2ca02c", label="Hybrid Computation Speedup ($S_{comp}$)")
    ax1.plot(units, speedup_e2e, marker="x", linestyle=":", linewidth=1.8, color="#d62728", label="End-to-End Speedup ($S_{e2e}$)")
    max_u = max(units.max(), 8)
    ax1.plot([1, max_u], [1, max_u], linestyle="--", color="gray", label="Ideal Linear Speedup", alpha=0.7)
    ax1.set_xlabel("Processing Units ($P = \\text{MPI Ranks} \\times \\text{OpenMP Threads}$)")
    ax1.set_ylabel("Speedup ($S = T_{seq} / T_{hybrid}$)")
    ax1.set_title("Hybrid MPI + OpenMP Strong Scaling Speedup")
    ax1.legend(loc="upper left", frameon=True)

    # Efficiency
    ax2.plot(units, efficiency_comp, marker="s", linewidth=2.4, color="#1f77b4", label="Hybrid Parallel Efficiency (%)")
    ax2.axhline(100.0, linestyle="--", color="gray", label="100% Ideal Efficiency", alpha=0.7)
    ax2.set_xlabel("Processing Units ($P = \\text{MPI Ranks} \\times \\text{OpenMP Threads}$)")
    ax2.set_ylabel("Parallel Efficiency (%)")
    ax2.set_title("Hybrid Parallel Computational Efficiency")
    ax2.set_ylim(0, 120)
    ax2.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig5_speedup_and_efficiency.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Visualizations for Hybrid MPI+OpenMP Analysis")
    parser.add_argument("--results-dir", default="results/hybrid", help="Directory containing CSV outputs")
    parser.add_argument("--benchmarks-file", default="results/benchmark_results.csv", help="Path to benchmark results CSV")
    parser.add_argument("--output-dir", default="results/plots", help="Directory to save generated PNG plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating charts from: {args.results_dir} ...")

    hourly_csv = os.path.join(args.results_dir, "hourly_stats.csv")
    daily_csv = os.path.join(args.results_dir, "daily_stats.csv")
    anomalies_csv = os.path.join(args.results_dir, "anomalies.csv")
    summary_csv = os.path.join(args.results_dir, "summary.csv")

    plot_hourly_profile(hourly_csv, args.output_dir)
    plot_daily_consumption(daily_csv, args.output_dir)
    plot_anomalies_distribution(anomalies_csv, summary_csv, args.output_dir)
    plot_benchmarks(args.benchmarks_file, args.output_dir)

    print(f"[+] All visualization plots generated successfully in: {args.output_dir}/")

if __name__ == "__main__":
    main()
