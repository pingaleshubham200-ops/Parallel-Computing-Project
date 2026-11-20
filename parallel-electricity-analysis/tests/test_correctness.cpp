#include "common.h"
#include "dataset_loader.h"
#include "statistics.h"
#include "sequential.h"
#include "hybrid_mpi_omp.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <mpi.h>
#include <omp.h>

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[-] ASSERTION FAILED: " << msg << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
            tests_failed++; \
            return; \
        } \
    } while(0)

#define ASSERT_NEAR(val1, val2, tol, msg) \
    do { \
        if (std::abs((val1) - (val2)) > (tol)) { \
            std::cerr << "[-] ASSERTION FAILED: " << msg << " | Expected " << (val2) << " +/- " << (tol) << ", got " << (val1) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            tests_failed++; \
            return; \
        } \
    } while(0)

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

void test_missing_values_and_malformed() {
    tests_run++;
    std::string test_path = "/tmp/test_smart_grid_missing.csv";
    {
        std::ofstream out(test_path);
        out << "Timestamp,Voltage (V),Current (A),Power Consumption (kW),Reactive Power (kVAR),Power Factor,Solar Power (kW),Wind Power (kW),Grid Supply (kW),Voltage Fluctuation (%),Overload Condition,Transformer Fault,Temperature (°C),Humidity (%),Electricity Price (USD/kWh),Predicted Load (kW)\n";
        out << "2024-01-01 00:00:00,230.5,10.2,2.345,0.4,0.95,0.0,5.0,0.0,0.2,0,0,18.5,50.0,0.15,2.30\n"; // Valid 1
        out << "2024-01-01 00:15:00,231.0,15.1,?,0.5,0.92,0.0,4.5,0.0,0.4,0,0,18.2,52.0,0.15,3.40\n";     // Rejected ('?')
        out << "2024-01-01 00:30:00,229.8,22.0,5.060,1.2,0.85,0.0,6.2,0.0,-0.1,1,0,17.9,55.0,0.18,5.10\n"; // Valid 2
        out << "bad_timestamp,230.0,10.0,2.300,0.4,0.90,0.0,5.0,0.0,0.0,0,0,20.0,50.0,0.15,2.30\n";        // Rejected (bad timestamp)
        out << "2024-01-01 00:45:00,230.0,10.0,-3.500,0.4,0.90,0.0,5.0,0.0,0.0,0,0,20.0,50.0,0.15,2.30\n"; // Rejected (negative power)
        out << "2024-01-01 01:00:00,228.5,45.2,10.35,2.1,0.91,0.0,8.0,0.0,-0.6,1,1,16.5,60.0,0.22,10.20\n"; // Valid 3
    }

    auto res = DatasetLoader::load_from_file(test_path);
    std::filesystem::remove(test_path);

    ASSERT_TRUE(res.success, "DatasetLoader should succeed on valid file with some rejected lines");
    ASSERT_TRUE(res.valid_records == 3, "Should have exactly 3 valid records");
    ASSERT_TRUE(res.rejected_records == 3, "Should have exactly 3 rejected records");
    ASSERT_TRUE(res.records.size() == 3, "Records vector size must equal valid records");

    tests_passed++;
    std::cout << "[+] test_missing_values_and_malformed passed.\n";
}

void test_timestamp_parsing() {
    tests_run++;
    int64_t epoch = DatasetLoader::to_epoch_seconds(2024, 1, 1, 0, 0, 0);
    ASSERT_TRUE(epoch == 1704067200LL, "Epoch for 2024-01-01 00:00:00 must equal 1704067200");

    int64_t epoch15 = DatasetLoader::to_epoch_seconds(2024, 1, 1, 0, 15, 0);
    ASSERT_TRUE(epoch15 == 1704068100LL, "15-minute interval epoch calculation must be exact");

    int64_t leap_epoch = DatasetLoader::to_epoch_seconds(2024, 2, 29, 0, 0, 0);
    ASSERT_TRUE(leap_epoch == 1709164800LL, "Leap day 2024-02-29 calculation must be exact");

    tests_passed++;
    std::cout << "[+] test_timestamp_parsing passed.\n";
}

void test_statistical_math() {
    tests_run++;
    std::vector<PowerReading> readings;
    for (int i = 1; i <= 4; ++i) {
        PowerReading r;
        std::memset(&r, 0, sizeof(r));
        r.timestamp_sec = 1704067200 + (i - 1) * 900;
        r.year = 2024; r.month = 1; r.day = 1;
        r.hour = 10; r.minute = (i - 1) * 15; r.second = 0;
        r.power_consumption = static_cast<double>(i * 2);
        r.solar_power = 2.0;
        r.wind_power = 3.0;
        r.electricity_price = 0.20;
        r.voltage = 230.0;
        r.power_factor = 0.95;
        snprintf(r.date_str, sizeof(r.date_str), "2024-01-01");
        snprintf(r.time_str, sizeof(r.time_str), "10:%02d:00", (i - 1) * 15);
        readings.push_back(r);
    }

    auto hourly = Statistics::compute_hourly(readings, 0.25);
    auto daily = Statistics::compute_daily(readings, 0.25);
    auto summary = Statistics::compute_global_summary(readings, 4, 0, 15.0, hourly, daily, 1.8);

    ASSERT_NEAR(summary.mean_active_power, 5.0, 1e-9, "Mean should be 5.0 kW");
    ASSERT_NEAR(summary.variance_active_power, 5.0, 1e-9, "Variance should be 5.0");
    ASSERT_NEAR(summary.stddev_active_power, std::sqrt(5.0), 1e-9, "StdDev should be sqrt(5.0)");
    ASSERT_NEAR(summary.min_active_power, 2.0, 1e-9, "Min power should be 2.0 kW");
    ASSERT_NEAR(summary.max_active_power, 8.0, 1e-9, "Max power should be 8.0 kW");
    ASSERT_NEAR(summary.total_energy_kwh, 5.0, 1e-9, "Total energy should be 5.0 kWh");
    ASSERT_NEAR(summary.total_renewable_energy_kwh, 5.0, 1e-9, "Total renewable energy should be 5.0 kWh");

    tests_passed++;
    std::cout << "[+] test_statistical_math passed.\n";
}

void test_anomaly_detection_rule() {
    tests_run++;
    std::vector<PowerReading> readings;
    for (int i = 0; i < 50; ++i) {
        PowerReading r;
        std::memset(&r, 0, sizeof(r));
        r.timestamp_sec = 1704067200 + i * 900;
        r.year = 2024; r.month = 1; r.day = 1;
        r.hour = (i * 15 / 60) % 24;
        r.minute = (i * 15) % 60;
        r.second = 0;
        r.power_consumption = 5.0;
        r.voltage = 230.0;
        r.overload_condition = 0;
        r.transformer_fault = 0;
        snprintf(r.date_str, sizeof(r.date_str), "2024-01-01");
        snprintf(r.time_str, sizeof(r.time_str), "%02d:%02d:00", r.hour, r.minute);
        readings.push_back(r);
    }

    // Add peak spike
    PowerReading spike;
    std::memset(&spike, 0, sizeof(spike));
    spike.timestamp_sec = 1704067200 + 50 * 900;
    spike.year = 2024; spike.month = 1; spike.day = 1;
    spike.hour = 12; spike.minute = 30; spike.second = 0;
    spike.power_consumption = 12.0;
    spike.voltage = 225.0;
    spike.overload_condition = 1;
    spike.transformer_fault = 0;
    snprintf(spike.date_str, sizeof(spike.date_str), "2024-01-01");
    snprintf(spike.time_str, sizeof(spike.time_str), "12:30:00");
    readings.push_back(spike);

    auto hourly = Statistics::compute_hourly(readings, 0.25);
    auto daily = Statistics::compute_daily(readings, 0.25);
    auto summary = Statistics::compute_global_summary(readings, 51, 0, 15.0, hourly, daily, 1.8);

    auto anomalies = Statistics::detect_anomalies(readings, summary.mean_active_power, summary.stddev_active_power, summary.anomaly_threshold);

    ASSERT_TRUE(anomalies.size() >= 1, "Should flag the spike reading");
    ASSERT_NEAR(anomalies[0].active_power, 12.0, 1e-9, "Flagged spike should be 12.0 kW");
    ASSERT_TRUE(anomalies[0].is_overload == 1, "Should record overload flag");

    tests_passed++;
    std::cout << "[+] test_anomaly_detection_rule passed.\n";
}

void test_sequential_vs_hybrid_consistency(int argc, char** argv) {
    tests_run++;
    std::string test_data_path = "/tmp/test_sg_hybrid_compare.csv";
    DatasetLoader::generate_sample_dataset(test_data_path, 3); // 3 days

    // 1. Run Sequential Reference
    auto seq_res = SequentialAnalysis::run(test_data_path, "", -1, 1.8);
    ASSERT_TRUE(seq_res.success, "Sequential analysis must succeed");

    // 2. Run Hybrid MPI + OpenMP
    auto hybrid_res = HybridAnalysis::run(argc, argv, test_data_path, "", PartitionStrategy::EQUAL_CHUNKS, -1, 1.8, true);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        ASSERT_TRUE(hybrid_res.success, "Hybrid analysis must succeed");

        // Verify valid records
        ASSERT_TRUE(seq_res.summary.valid_records == hybrid_res.summary.valid_records, "Record count mismatch");
        ASSERT_TRUE(seq_res.summary.rejected_records == hybrid_res.summary.rejected_records, "Rejected count mismatch");

        // Verify Mean active power
        ASSERT_NEAR(seq_res.summary.mean_active_power, hybrid_res.summary.mean_active_power, 1e-6, "Mean mismatch");

        // Verify StdDev active power
        ASSERT_NEAR(seq_res.summary.stddev_active_power, hybrid_res.summary.stddev_active_power, 1e-6, "StdDev mismatch");

        // Verify Peak power & timestamp
        ASSERT_NEAR(seq_res.summary.peak_demand_power, hybrid_res.summary.peak_demand_power, 1e-6, "Peak demand mismatch");
        ASSERT_TRUE(seq_res.summary.peak_timestamp_str == hybrid_res.summary.peak_timestamp_str, "Peak timestamp mismatch");

        // Verify Energy and Cost
        ASSERT_NEAR(seq_res.summary.total_energy_kwh, hybrid_res.summary.total_energy_kwh, 1e-6, "Energy mismatch");
        ASSERT_NEAR(seq_res.summary.total_cost_usd, hybrid_res.summary.total_cost_usd, 1e-6, "Cost mismatch");
        ASSERT_NEAR(seq_res.summary.total_renewable_energy_kwh, hybrid_res.summary.total_renewable_energy_kwh, 1e-6, "Renewable energy mismatch");

        // Verify Anomaly Count
        ASSERT_TRUE(seq_res.summary.anomaly_count == hybrid_res.summary.anomaly_count, "Anomaly count mismatch");

        // Verify Hourly Aggregations
        for (int h = 0; h < 24; ++h) {
            ASSERT_TRUE(seq_res.hourly[h].count == hybrid_res.hourly[h].count, "Hourly count mismatch");
            if (seq_res.hourly[h].count > 0) {
                ASSERT_NEAR(seq_res.hourly[h].mean(), hybrid_res.hourly[h].mean(), 1e-6, "Hourly mean mismatch");
                ASSERT_NEAR(seq_res.hourly[h].max_power, hybrid_res.hourly[h].max_power, 1e-6, "Hourly max power mismatch");
            }
        }

        tests_passed++;
        std::cout << "[+] test_sequential_vs_hybrid_consistency passed (Sequential Baseline == Hybrid MPI+OpenMP).\n";
    }

    std::filesystem::remove(test_data_path);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "=================================================================\n";
        std::cout << "  CORRECTNESS TEST SUITE: Hybrid MPI + OpenMP Analysis Engine    \n";
        std::cout << "=================================================================\n";

        test_missing_values_and_malformed();
        test_timestamp_parsing();
        test_statistical_math();
        test_anomaly_detection_rule();
    }

    test_sequential_vs_hybrid_consistency(argc, argv);

    if (rank == 0) {
        std::cout << "=================================================================\n";
        std::cout << " Test Summary: " << tests_passed << "/" << tests_run << " passed. "
                  << "Failures: " << tests_failed << "\n";
        std::cout << "=================================================================\n";
    }

    MPI_Finalize();
    return (tests_failed == 0) ? 0 : 1;
}
