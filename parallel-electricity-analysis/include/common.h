#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <map>

// Data structure representing a single smart grid telemetry reading.
// Designed with POD layout for direct memory mapping and MPI_Datatype packing.
struct PowerReading {
    int64_t timestamp_sec;      // Unix epoch timestamp (seconds)
    int year;
    int month;
    int day;
    int hour;
    int minute;
    int second;

    double power_consumption;   // Active load demand (kW) - Primary analysis field
    double voltage;             // Line voltage (V)
    double current;             // Current (A)
    double reactive_power;      // Reactive power (kVAR)
    double power_factor;        // Power factor (cos phi)
    double solar_power;         // Renewable solar generation (kW)
    double wind_power;          // Renewable wind generation (kW)
    double grid_supply;         // Grid import/export (kW)
    double voltage_fluctuation; // Voltage fluctuation (%)
    int overload_condition;     // 1 if overload alert active, 0 otherwise
    int transformer_fault;      // 1 if transformer fault active, 0 otherwise
    double temperature;         // Ambient temperature (°C)
    double humidity;            // Relative humidity (%)
    double electricity_price;   // Dynamic tariff ($/kWh)
    double predicted_load;      // ML baseline forecasted load (kW)

    char date_str[12];          // "YYYY-MM-DD\0"
    char time_str[10];          // "HH:MM:SS\0"
};

// Statistical summary for a time window (hourly or daily)
struct AggregationStats {
    int64_t count = 0;
    double sum_power = 0.0;
    double sum_power_sq = 0.0;
    double min_power = 1e9;
    double max_power = -1e9;

    double sum_solar = 0.0;
    double sum_wind = 0.0;
    double sum_cost = 0.0;
    int64_t overload_count = 0;
    int64_t fault_count = 0;

    double mean() const {
        return (count > 0) ? (sum_power / count) : 0.0;
    }

    double variance() const {
        if (count <= 1) return 0.0;
        double m = mean();
        double var = (sum_power_sq / count) - (m * m);
        return (var < 0.0) ? 0.0 : var;
    }

    double std_dev() const {
        return (variance() > 0.0) ? std::sqrt(variance()) : 0.0;
    }

    // Smart grid 15-minute sampling interval = 15/60 = 0.25 hours
    // Energy (kWh) = sum(P (kW)) * 0.25 hr
    double total_energy_kwh(double interval_hours = 0.25) const {
        return sum_power * interval_hours;
    }

    double total_renewable_energy_kwh(double interval_hours = 0.25) const {
        return (sum_solar + sum_wind) * interval_hours;
    }
};

// Overall dataset summary statistics
struct GlobalSummary {
    int64_t total_records_read = 0;
    int64_t valid_records = 0;
    int64_t rejected_records = 0;

    double sampling_interval_minutes = 15.0; // 15-minute intervals in smart grid dataset
    double sampling_interval_hours = 0.25;

    // Power consumption statistics (kW)
    double mean_active_power = 0.0;
    double variance_active_power = 0.0;
    double stddev_active_power = 0.0;
    double min_active_power = 0.0;
    double max_active_power = 0.0;
    double total_energy_kwh = 0.0;
    double total_cost_usd = 0.0;

    // Renewable generation
    double total_solar_energy_kwh = 0.0;
    double total_wind_energy_kwh = 0.0;
    double total_renewable_energy_kwh = 0.0;

    // Grid reliability metrics
    int64_t total_overload_intervals = 0;
    int64_t total_transformer_faults = 0;
    double avg_voltage = 0.0;
    double avg_power_factor = 0.0;

    // Peak demand info
    double peak_demand_power = 0.0;
    std::string peak_timestamp_str;
    int64_t peak_timestamp_sec = 0;

    // Peak periods
    int peak_hour_avg = -1;
    double peak_hour_avg_power = 0.0;
    int peak_hour_max = -1;
    double peak_hour_max_power = 0.0;

    std::string peak_day_avg;
    double peak_day_avg_power = 0.0;
    std::string peak_day_max;
    double peak_day_max_power = 0.0;

    // Anomaly detection
    double anomaly_threshold = 0.0;
    double anomaly_k_factor = 1.8; // Configurable k-factor (default 1.8 flags upper tails)
    int64_t anomaly_count = 0;
};

// Represents a flagged anomaly or grid event
struct AnomalyRecord {
    std::string timestamp_str;
    int64_t timestamp_sec;
    double active_power;        // kW
    double z_score;             // (value - mean) / stddev
    int is_overload;            // 1 if overload condition active
    int is_fault;               // 1 if transformer fault active
    double voltage;             // V
    std::string alert_category; // "STATISTICAL_SPIKE", "GRID_OVERLOAD", or "TRANSFORMER_FAULT"
};

// Execution timing breakdown (seconds)
struct ExecutionTiming {
    double t_load_preproc = 0.0;
    double t_distribution = 0.0;
    double t_computation = 0.0;
    double t_aggregation = 0.0;
    double t_anomaly_detect = 0.0;
    double t_parallel_total = 0.0;
    double t_end_to_end = 0.0;

    int mpi_ranks = 1;
    int omp_threads = 1;
    std::string partitioning_strategy;
};

// Partitioning strategy enum
enum class PartitionStrategy {
    EQUAL_CHUNKS,      // Block distribution: ~N / P contiguous records per rank
    TIME_RANGE         // Chronological time interval based distribution: [T_start, T_end] divided into P intervals
};
