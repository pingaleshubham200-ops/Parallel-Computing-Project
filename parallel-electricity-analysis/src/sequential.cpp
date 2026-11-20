#include "sequential.h"
#include "dataset_loader.h"
#include "statistics.h"
#include <chrono>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

SequentialAnalysis::Result SequentialAnalysis::run(
    const std::string& input_filepath,
    const std::string& output_dir,
    int64_t max_records,
    double anomaly_k_factor
) {
    Result res;
    auto t_start_total = std::chrono::high_resolution_clock::now();

    // 1. Ingestion & Preprocessing
    auto t_start_load = std::chrono::high_resolution_clock::now();
    auto load_res = DatasetLoader::load_from_file(input_filepath, max_records);
    auto t_end_load = std::chrono::high_resolution_clock::now();

    if (!load_res.success) {
        res.success = false;
        res.error_message = load_res.error_message;
        return res;
    }

    res.timing.t_load_preproc = std::chrono::duration<double>(t_end_load - t_start_load).count();
    res.timing.mpi_ranks = 1;
    res.timing.omp_threads = 1;
    res.timing.partitioning_strategy = "Sequential (Single-Threaded)";

    double interval_hours = load_res.detected_interval_minutes / 60.0;

    // 2. Statistical Computations
    auto t_start_comp = std::chrono::high_resolution_clock::now();
    res.hourly = Statistics::compute_hourly(load_res.records, interval_hours);
    res.daily = Statistics::compute_daily(load_res.records, interval_hours);
    res.summary = Statistics::compute_global_summary(
        load_res.records,
        load_res.total_lines_read,
        load_res.rejected_records,
        load_res.detected_interval_minutes,
        res.hourly,
        res.daily,
        anomaly_k_factor
    );
    auto t_end_comp = std::chrono::high_resolution_clock::now();
    res.timing.t_computation = std::chrono::duration<double>(t_end_comp - t_start_comp).count();

    // 3. Anomaly Detection
    auto t_start_anom = std::chrono::high_resolution_clock::now();
    res.anomalies = Statistics::detect_anomalies(
        load_res.records,
        res.summary.mean_active_power,
        res.summary.stddev_active_power,
        res.summary.anomaly_threshold
    );
    res.summary.anomaly_count = static_cast<int64_t>(res.anomalies.size());
    auto t_end_anom = std::chrono::high_resolution_clock::now();
    res.timing.t_anomaly_detect = std::chrono::duration<double>(t_end_anom - t_start_anom).count();

    auto t_end_total = std::chrono::high_resolution_clock::now();
    res.timing.t_end_to_end = std::chrono::duration<double>(t_end_total - t_start_total).count();
    res.timing.t_parallel_total = res.timing.t_computation + res.timing.t_anomaly_detect;

    // 4. Save results to CSV
    if (!output_dir.empty()) {
        try {
            fs::create_directories(output_dir);
            Statistics::export_summary_csv(output_dir + "/summary.csv", res.summary, res.timing);
            Statistics::export_hourly_csv(output_dir + "/hourly_stats.csv", res.hourly, interval_hours);
            Statistics::export_daily_csv(output_dir + "/daily_stats.csv", res.daily, interval_hours);
            Statistics::export_anomalies_csv(output_dir + "/anomalies.csv", res.anomalies);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to write output files: " << e.what() << "\n";
        }
    }

    res.success = true;
    return res;
}
