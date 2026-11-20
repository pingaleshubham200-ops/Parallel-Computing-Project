#!/usr/bin/env python3
"""
Automated benchmark runner for Hybrid MPI + OpenMP Electricity Analysis:
Compares the Sequential Reference baseline against multi-core Hybrid MPI + OpenMP configurations.
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
import time

def parse_summary_csv(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return dict(zip(df["Metric"], df["Value"]))

def run_command(cmd, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    res = subprocess.run(cmd, shell=True, env=merged_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res

def main():
    parser = argparse.ArgumentParser(description="Benchmarking suite for Hybrid MPI + OpenMP Smart Grid Load Analysis")
    parser.add_argument("--input", default="data/smart_grid_dataset.csv", help="Input dataset path")
    parser.add_argument("--output-dir", default="results", help="Directory for benchmark outputs")
    parser.add_argument("--runs", type=int, default=2, help="Number of repetitions per configuration to average")
    parser.add_argument("--k-factor", type=float, default=1.8, help="Anomaly k-factor multiplier")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    benchmarks = []

    print("==========================================================================")
    print("      HYBRID MPI + OpenMP PARALLEL PERFORMANCE BENCHMARKING SUITE         ")
    print(f" Dataset: {args.input} | Repetitions: {args.runs} per configuration")
    print("==========================================================================")

    # 1. Sequential Reference Baseline (1 Process x 1 Thread)
    print("\n[*] Evaluating Sequential Baseline (1 Process x 1 Thread Reference)...")
    seq_out = os.path.join(args.output_dir, "benchmark_seq")
    seq_times = []
    seq_metrics = None

    for r in range(args.runs):
        t0 = time.time()
        cmd = f"./build/sequential --input {args.input} --output {seq_out} --k-factor {args.k_factor}"
        res = run_command(cmd)
        elapsed = time.time() - t0
        if res.returncode != 0:
            print(f"[-] Sequential run failed:\n{res.stderr}")
            sys.exit(1)
        seq_times.append(elapsed)
        seq_metrics = parse_summary_csv(os.path.join(seq_out, "summary.csv"))

    t_seq_avg = sum(seq_times) / len(seq_times)
    t_load_seq = float(seq_metrics.get("Time_Data_Loading", 0.0))
    t_comp_seq = float(seq_metrics.get("Time_Computation", 0.0))
    t_anom_seq = float(seq_metrics.get("Time_Anomaly_Detection", 0.0))
    t_comp_total_seq = t_comp_seq + t_anom_seq

    benchmarks.append({
        "Config_Name": "Sequential (1P x 1T)",
        "Paradigm": "Sequential Baseline",
        "MPI_Ranks": 1,
        "OMP_Threads": 1,
        "Total_Units": 1,
        "Strategy": "Single-Threaded",
        "T_Load_s": t_load_seq,
        "T_Dist_s": 0.0,
        "T_Comp_s": t_comp_total_seq,
        "T_Comm_s": 0.0,
        "T_Parallel_s": t_comp_total_seq,
        "T_EndToEnd_s": t_seq_avg,
        "Speedup_E2E": 1.0,
        "Efficiency_E2E": 1.0,
        "Speedup_Comp": 1.0,
        "Efficiency_Comp": 1.0
    })
    print(f"    Avg Wall Time: {t_seq_avg:.4f} s | Comp Time: {t_comp_total_seq:.4f} s | Speedup: 1.00x")

    # 2. Hybrid MPI + OpenMP Test Configurations
    hybrid_configs = [
        {"ranks": 2, "threads": 2, "strategy": "equal-chunks"},
        {"ranks": 2, "threads": 2, "strategy": "time-range"},
        {"ranks": 2, "threads": 4, "strategy": "equal-chunks"},
        {"ranks": 4, "threads": 2, "strategy": "equal-chunks"},
        {"ranks": 4, "threads": 2, "strategy": "time-range"}
    ]

    for cfg in hybrid_configs:
        ranks = cfg["ranks"]
        threads = cfg["threads"]
        strategy = cfg["strategy"]
        P = ranks * threads
        cfg_name = f"Hybrid ({ranks}P x {threads}T, {strategy})"
        print(f"\n[*] Evaluating Hybrid MPI+OpenMP: {cfg_name}...")

        cfg_out = os.path.join(args.output_dir, f"bench_hybrid_{ranks}p_{threads}t_{strategy}")
        run_times = []
        metrics = None

        for r in range(args.runs):
            t0 = time.time()
            env = {"OMP_NUM_THREADS": str(threads)}
            cmd = f"mpirun -np {ranks} ./build/hybrid --input {args.input} --output {cfg_out} --strategy {strategy} --k-factor {args.k_factor}"
            res = run_command(cmd, env)
            elapsed = time.time() - t0
            if res.returncode != 0:
                print(f"[-] Hybrid run failed for {cfg_name}:\n{res.stderr}")
                continue
            run_times.append(elapsed)
            metrics = parse_summary_csv(os.path.join(cfg_out, "summary.csv"))

        if not run_times or not metrics:
            continue

        t_avg = sum(run_times) / len(run_times)
        speedup_e2e = t_seq_avg / t_avg if t_avg > 0 else 1.0
        efficiency_e2e = (speedup_e2e / P) if P > 0 else 1.0

        t_load = float(metrics.get("Time_Data_Loading", 0.0))
        t_dist = float(metrics.get("Time_Data_Distribution", 0.0))
        t_comp = float(metrics.get("Time_Computation", 0.0))
        t_comm = float(metrics.get("Time_MPI_Aggregation", 0.0))
        t_anom = float(metrics.get("Time_Anomaly_Detection", 0.0))
        t_par = float(metrics.get("Time_Parallel_Phase", 0.0))

        t_comp_total_par = t_comp + t_anom
        speedup_comp = (t_comp_total_seq / t_comp_total_par) if t_comp_total_par > 0 else 1.0
        efficiency_comp = (speedup_comp / P) if P > 0 else 1.0

        benchmarks.append({
            "Config_Name": cfg_name,
            "Paradigm": "Hybrid MPI+OpenMP",
            "MPI_Ranks": ranks,
            "OMP_Threads": threads,
            "Total_Units": P,
            "Strategy": strategy,
            "T_Load_s": t_load,
            "T_Dist_s": t_dist,
            "T_Comp_s": t_comp_total_par,
            "T_Comm_s": t_comm,
            "T_Parallel_s": t_par,
            "T_EndToEnd_s": t_avg,
            "Speedup_E2E": speedup_e2e,
            "Efficiency_E2E": efficiency_e2e,
            "Speedup_Comp": speedup_comp,
            "Efficiency_Comp": efficiency_comp
        })
        print(f"    Avg Wall Time: {t_avg:.4f} s | Comp Speedup: {speedup_comp:.2f}x | Comp Eff: {efficiency_comp*100.0:.1f}%")

    bench_df = pd.DataFrame(benchmarks)
    csv_file = os.path.join(args.output_dir, "benchmark_results.csv")
    bench_df.to_csv(csv_file, index=False)
    print("\n==========================================================================")
    print(f"[+] Hybrid MPI + OpenMP Benchmarks completed! Saved to: {csv_file}")
    print("==========================================================================")

if __name__ == "__main__":
    main()
