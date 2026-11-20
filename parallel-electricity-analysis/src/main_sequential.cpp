#include "sequential.h"
#include "statistics.h"
#include "dataset_loader.h"
#include <iostream>
#include <string>

void print_help(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --input <path>         Path to input dataset CSV (default: data/smart_grid_dataset.csv)\n"
              << "  --output <dir>         Directory to write output CSV files (default: results/sequential)\n"
              << "  --max-records <N>      Limit processing to first N valid records (-1 for all, default: -1)\n"
              << "  --k-factor <K>         Anomaly threshold multiplier K for Mean + K*StdDev (default: 1.8)\n"
              << "  --generate-sample      Generate synthetic sample dataset and exit\n"
              << "  --help, -h             Display this help message and exit\n\n"
              << "Example:\n"
              << "  " << prog << " --input data/smart_grid_dataset.csv --output results/sequential\n";
}

int main(int argc, char** argv) {
    std::string input_path = "data/smart_grid_dataset.csv";
    std::string output_dir = "results/sequential";
    int64_t max_records = -1;
    double anomaly_k_factor = 1.8;
    bool gen_sample = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--max-records" && i + 1 < argc) {
            max_records = std::stoll(argv[++i]);
        } else if (arg == "--k-factor" && i + 1 < argc) {
            anomaly_k_factor = std::stod(argv[++i]);
        } else if (arg == "--generate-sample") {
            gen_sample = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_help(argv[0]);
            return 1;
        }
    }

    if (gen_sample) {
        std::cout << "Generating synthetic sample dataset to: " << input_path << " ...\n";
        if (DatasetLoader::generate_sample_dataset(input_path, 14)) {
            std::cout << "Successfully generated sample dataset (14 days = 1,344 readings)!\n";
            return 0;
        } else {
            std::cerr << "Failed to generate sample dataset.\n";
            return 1;
        }
    }

    std::cout << "Running Sequential Smart Grid Load & Peak Analysis...\n";
    std::cout << "  Input dataset: " << input_path << "\n";
    std::cout << "  Output dir   : " << output_dir << "\n";

    auto res = SequentialAnalysis::run(input_path, output_dir, max_records, anomaly_k_factor);
    if (!res.success) {
        std::cerr << "\nError: " << res.error_message << "\n";
        return 1;
    }

    Statistics::print_summary_report(res.summary, res.timing);
    std::cout << "[+] Sequential analysis complete. Results exported to: " << output_dir << "/\n";

    return 0;
}
