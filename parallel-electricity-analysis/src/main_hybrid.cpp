#include "hybrid_mpi_omp.h"
#include "statistics.h"
#include "dataset_loader.h"
#include <iostream>
#include <string>
#include <mpi.h>

void print_help(const char* prog) {
    std::cout << "Usage: mpirun -np <processes> " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --input <path>                    Path to input dataset CSV (default: data/smart_grid_dataset.csv)\n"
              << "  --output <dir>                    Directory to write output CSV files (default: results/hybrid)\n"
              << "  --strategy <equal-chunks|time-range> Data partitioning strategy (default: equal-chunks)\n"
              << "  --max-records <N>                 Limit processing to first N valid records (-1 for all, default: -1)\n"
              << "  --k-factor <K>                    Anomaly threshold multiplier K for Mean + K*StdDev (default: 1.8)\n"
              << "  --help, -h                        Display this help message and exit\n\n"
              << "Example:\n"
              << "  OMP_NUM_THREADS=2 mpirun -np 4 " << prog << " --input data/smart_grid_dataset.csv --output results/hybrid\n";
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string input_path = "data/smart_grid_dataset.csv";
    std::string output_dir = "results/hybrid";
    PartitionStrategy strategy = PartitionStrategy::EQUAL_CHUNKS;
    int64_t max_records = -1;
    double anomaly_k_factor = 1.8;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            if (rank == 0) print_help(argv[0]);
            MPI_Finalize();
            return 0;
        } else if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--strategy" && i + 1 < argc) {
            std::string s = argv[++i];
            if (s == "time-range" || s == "time_range") {
                strategy = PartitionStrategy::TIME_RANGE;
            } else if (s == "equal-chunks" || s == "equal_chunks") {
                strategy = PartitionStrategy::EQUAL_CHUNKS;
            } else {
                if (rank == 0) std::cerr << "Unknown strategy: " << s << ". Using equal-chunks.\n";
            }
        } else if (arg == "--max-records" && i + 1 < argc) {
            max_records = std::stoll(argv[++i]);
        } else if (arg == "--k-factor" && i + 1 < argc) {
            anomaly_k_factor = std::stod(argv[++i]);
        } else {
            if (rank == 0) {
                std::cerr << "Unknown argument: " << arg << "\n";
                print_help(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        std::cout << "Running Hybrid MPI + OpenMP Smart Grid Load & Peak Analysis...\n";
        std::cout << "  Input dataset: " << input_path << "\n";
        std::cout << "  Output dir   : " << output_dir << "\n";
        std::cout << "  Strategy     : " << ((strategy == PartitionStrategy::EQUAL_CHUNKS) ? "equal-chunks" : "time-range") << "\n";
    }

    auto res = HybridAnalysis::run(argc, argv, input_path, output_dir, strategy, max_records, anomaly_k_factor, true);

    if (rank == 0) {
        if (!res.success) {
            std::cerr << "\nError: " << res.error_message << "\n";
            MPI_Finalize();
            return 1;
        }

        Statistics::print_summary_report(res.summary, res.timing);
        std::cout << "[+] Hybrid analysis complete. Results exported to: " << output_dir << "/\n";
    }

    MPI_Finalize();
    return 0;
}
