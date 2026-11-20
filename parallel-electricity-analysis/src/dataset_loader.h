#pragma once

#include "common.h"
#include <string>
#include <vector>

class DatasetLoader {
public:
    struct LoadResult {
        std::vector<PowerReading> records;
        int64_t total_lines_read = 0;
        int64_t valid_records = 0;
        int64_t rejected_records = 0;
        double detected_interval_minutes = 15.0;
        bool success = false;
        std::string error_message;
    };

    // Load dataset from CSV (Smart Grid Kaggle dataset or UCI format)
    static LoadResult load_from_file(const std::string& filepath, int64_t max_records = -1);

    // Fast civil calendar to UTC Unix epoch converter
    static int64_t to_epoch_seconds(int year, int month, int day, int hour, int minute, int second);

    // Synthetic generator for verification and unit tests
    static bool generate_sample_dataset(const std::string& output_path, int num_days = 7);
};
