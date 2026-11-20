#pragma once

#include "common.h"
#include <vector>
#include <map>
#include <string>

class Statistics {
public:
    // Compute hourly aggregation (hours 0..23)
    static std::vector<AggregationStats> compute_hourly(const std::vector<PowerReading>& readings, double interval_hours = 0.25);

    // Compute daily aggregation (grouped by date string)
    static std::map<std::string, AggregationStats> compute_daily(const std::vector<PowerReading>& readings, double interval_hours = 0.25);

    // Compute global summary and peak-demand metrics
    static GlobalSummary compute_global_summary(
        const std::vector<PowerReading>& readings,
        int64_t total_lines_read,
        int64_t rejected_count,
        double interval_minutes,
        const std::vector<AggregationStats>& hourly,
        const std::map<std::string, AggregationStats>& daily,
        double anomaly_k_factor = 1.8
    );

    // Detect anomalies and critical grid events
    static std::vector<AnomalyRecord> detect_anomalies(
        const std::vector<PowerReading>& readings,
        double mean,
        double stddev,
        double threshold
    );

    // CSV Exporters
    static bool export_summary_csv(const std::string& filepath, const GlobalSummary& summary, const ExecutionTiming& timing);
    static bool export_hourly_csv(const std::string& filepath, const std::vector<AggregationStats>& hourly, double interval_hours = 0.25);
    static bool export_daily_csv(const std::string& filepath, const std::map<std::string, AggregationStats>& daily, double interval_hours = 0.25);
    static bool export_anomalies_csv(const std::string& filepath, const std::vector<AnomalyRecord>& anomalies);
    static bool export_benchmark_csv(const std::string& filepath, const std::vector<ExecutionTiming>& benchmark_records);

    // Console output formatter
    static void print_summary_report(const GlobalSummary& summary, const ExecutionTiming& timing);
};
