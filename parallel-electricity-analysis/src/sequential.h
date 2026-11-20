#pragma once

#include "common.h"
#include <string>

class SequentialAnalysis {
public:
    struct Result {
        GlobalSummary summary;
        std::vector<AggregationStats> hourly;
        std::map<std::string, AggregationStats> daily;
        std::vector<AnomalyRecord> anomalies;
        ExecutionTiming timing;
        bool success = false;
        std::string error_message;
    };

    static Result run(
        const std::string& input_filepath,
        const std::string& output_dir = "",
        int64_t max_records = -1,
        double anomaly_k_factor = 1.8
    );
};
