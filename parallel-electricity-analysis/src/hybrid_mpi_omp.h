#pragma once

#include "common.h"
#include <string>
#include <mpi.h>

class HybridAnalysis {
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

    // Run hybrid MPI + OpenMP distributed analysis pipeline
    static Result run(
        int argc,
        char** argv,
        const std::string& input_filepath,
        const std::string& output_dir = "",
        PartitionStrategy strategy = PartitionStrategy::EQUAL_CHUNKS,
        int64_t max_records = -1,
        double anomaly_k_factor = 1.8,
        bool mpi_already_initialized = false
    );

    // Create MPI_Datatype for PowerReading
    static MPI_Datatype create_mpi_reading_type();
};
