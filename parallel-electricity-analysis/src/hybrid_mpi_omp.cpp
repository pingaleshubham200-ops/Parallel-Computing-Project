#include "hybrid_mpi_omp.h"
#include "dataset_loader.h"
#include "statistics.h"
#include <omp.h>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;

struct DailyTransferEntry {
    char date_str[12];
    int64_t count;
    double sum_power;
    double sum_power_sq;
    double min_power;
    double max_power;
    double sum_solar;
    double sum_wind;
    double sum_cost;
    int64_t overload_count;
    int64_t fault_count;
};

struct AnomalyTransferEntry {
    char timestamp_str[24];
    int64_t timestamp_sec;
    double active_power;
    double voltage;
    double z_score;
    int is_overload;
    int is_fault;
    char alert_category[32];
};

struct PeakCandidate {
    double max_power;
    int64_t timestamp_sec;
    char date_str[12];
    char time_str[10];
};

MPI_Datatype HybridAnalysis::create_mpi_reading_type() {
    MPI_Datatype reading_type;
    int count = 24;
    int blocklengths[24] = {
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1,
        1, 1, 1, 1,
        12, 10
    };
    MPI_Aint displacements[24];
    MPI_Datatype types[24] = {
        MPI_INT64_T, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
        MPI_INT, MPI_INT,
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
        MPI_CHAR, MPI_CHAR
    };

    displacements[0]  = offsetof(PowerReading, timestamp_sec);
    displacements[1]  = offsetof(PowerReading, year);
    displacements[2]  = offsetof(PowerReading, month);
    displacements[3]  = offsetof(PowerReading, day);
    displacements[4]  = offsetof(PowerReading, hour);
    displacements[5]  = offsetof(PowerReading, minute);
    displacements[6]  = offsetof(PowerReading, second);
    displacements[7]  = offsetof(PowerReading, power_consumption);
    displacements[8]  = offsetof(PowerReading, voltage);
    displacements[9]  = offsetof(PowerReading, current);
    displacements[10] = offsetof(PowerReading, reactive_power);
    displacements[11] = offsetof(PowerReading, power_factor);
    displacements[12] = offsetof(PowerReading, solar_power);
    displacements[13] = offsetof(PowerReading, wind_power);
    displacements[14] = offsetof(PowerReading, grid_supply);
    displacements[15] = offsetof(PowerReading, voltage_fluctuation);
    displacements[16] = offsetof(PowerReading, overload_condition);
    displacements[17] = offsetof(PowerReading, transformer_fault);
    displacements[18] = offsetof(PowerReading, temperature);
    displacements[19] = offsetof(PowerReading, humidity);
    displacements[20] = offsetof(PowerReading, electricity_price);
    displacements[21] = offsetof(PowerReading, predicted_load);
    displacements[22] = offsetof(PowerReading, date_str);
    displacements[23] = offsetof(PowerReading, time_str);

    MPI_Type_create_struct(count, blocklengths, displacements, types, &reading_type);
    MPI_Type_commit(&reading_type);
    return reading_type;
}

static MPI_Datatype create_mpi_daily_type() {
    MPI_Datatype daily_type;
    int count = 11;
    int blocklengths[11] = {12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint displacements[11];
    MPI_Datatype types[11] = {
        MPI_CHAR, MPI_INT64_T, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT64_T, MPI_INT64_T
    };

    displacements[0]  = offsetof(DailyTransferEntry, date_str);
    displacements[1]  = offsetof(DailyTransferEntry, count);
    displacements[2]  = offsetof(DailyTransferEntry, sum_power);
    displacements[3]  = offsetof(DailyTransferEntry, sum_power_sq);
    displacements[4]  = offsetof(DailyTransferEntry, min_power);
    displacements[5]  = offsetof(DailyTransferEntry, max_power);
    displacements[6]  = offsetof(DailyTransferEntry, sum_solar);
    displacements[7]  = offsetof(DailyTransferEntry, sum_wind);
    displacements[8]  = offsetof(DailyTransferEntry, sum_cost);
    displacements[9]  = offsetof(DailyTransferEntry, overload_count);
    displacements[10] = offsetof(DailyTransferEntry, fault_count);

    MPI_Type_create_struct(count, blocklengths, displacements, types, &daily_type);
    MPI_Type_commit(&daily_type);
    return daily_type;
}

static MPI_Datatype create_mpi_anomaly_type_internal() {
    MPI_Datatype anomaly_type;
    int count = 8;
    int blocklengths[8] = {24, 1, 1, 1, 1, 1, 1, 32};
    MPI_Aint displacements[8];
    MPI_Datatype types[8] = {
        MPI_CHAR, MPI_INT64_T, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_CHAR
    };

    displacements[0] = offsetof(AnomalyTransferEntry, timestamp_str);
    displacements[1] = offsetof(AnomalyTransferEntry, timestamp_sec);
    displacements[2] = offsetof(AnomalyTransferEntry, active_power);
    displacements[3] = offsetof(AnomalyTransferEntry, voltage);
    displacements[4] = offsetof(AnomalyTransferEntry, z_score);
    displacements[5] = offsetof(AnomalyTransferEntry, is_overload);
    displacements[6] = offsetof(AnomalyTransferEntry, is_fault);
    displacements[7] = offsetof(AnomalyTransferEntry, alert_category);

    MPI_Type_create_struct(count, blocklengths, displacements, types, &anomaly_type);
    MPI_Type_commit(&anomaly_type);
    return anomaly_type;
}

static MPI_Datatype create_mpi_peak_candidate_type() {
    MPI_Datatype peak_type;
    int count = 4;
    int blocklengths[4] = {1, 1, 12, 10};
    MPI_Aint displacements[4];
    MPI_Datatype types[4] = {
        MPI_DOUBLE, MPI_INT64_T, MPI_CHAR, MPI_CHAR
    };

    displacements[0] = offsetof(PeakCandidate, max_power);
    displacements[1] = offsetof(PeakCandidate, timestamp_sec);
    displacements[2] = offsetof(PeakCandidate, date_str);
    displacements[3] = offsetof(PeakCandidate, time_str);

    MPI_Type_create_struct(count, blocklengths, displacements, types, &peak_type);
    MPI_Type_commit(&peak_type);
    return peak_type;
}

HybridAnalysis::Result HybridAnalysis::run(
    int argc,
    char** argv,
    const std::string& input_filepath,
    const std::string& output_dir,
    PartitionStrategy strategy,
    int64_t max_records,
    double anomaly_k_factor,
    bool mpi_already_initialized
) {
    Result res;
    int rank = 0, size = 1;
    bool finalized_mpi_locally = false;

    if (!mpi_already_initialized) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        finalized_mpi_locally = true;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    res.timing.mpi_ranks = size;
    res.timing.omp_threads = num_threads;
    res.timing.partitioning_strategy = (strategy == PartitionStrategy::EQUAL_CHUNKS) ? "Equal-Sized Chunks" : "Contiguous Time-Range";

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start_total = MPI_Wtime();

    MPI_Datatype mpi_reading_type = create_mpi_reading_type();
    MPI_Datatype mpi_daily_type = create_mpi_daily_type();
    MPI_Datatype mpi_anomaly_type = create_mpi_anomaly_type_internal();
    MPI_Datatype mpi_peak_type = create_mpi_peak_candidate_type();

    // 1. Data Ingestion (Rank 0)
    DatasetLoader::LoadResult load_res;
    double t_start_load = MPI_Wtime();
    if (rank == 0) {
        load_res = DatasetLoader::load_from_file(input_filepath, max_records);
    }
    double t_end_load = MPI_Wtime();
    if (rank == 0) {
        res.timing.t_load_preproc = t_end_load - t_start_load;
    }

    int load_success = (rank == 0 && load_res.success) ? 1 : 0;
    MPI_Bcast(&load_success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!load_success) {
        if (rank == 0) {
            res.success = false;
            res.error_message = load_res.error_message.empty() ? "File loading failed" : load_res.error_message;
        }
        MPI_Type_free(&mpi_reading_type);
        MPI_Type_free(&mpi_daily_type);
        MPI_Type_free(&mpi_anomaly_type);
        MPI_Type_free(&mpi_peak_type);
        if (finalized_mpi_locally) MPI_Finalize();
        return res;
    }

    // 2. Data Partitioning & Distribution
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start_dist = MPI_Wtime();

    int64_t total_valid_records = 0;
    int64_t total_lines_read = 0;
    int64_t total_rejected = 0;
    double detected_interval_minutes = 15.0;

    if (rank == 0) {
        total_valid_records = static_cast<int64_t>(load_res.records.size());
        total_lines_read = load_res.total_lines_read;
        total_rejected = load_res.rejected_records;
        detected_interval_minutes = load_res.detected_interval_minutes;
    }

    MPI_Bcast(&total_valid_records, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_lines_read, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_rejected, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&detected_interval_minutes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double interval_hours = detected_interval_minutes / 60.0;

    std::vector<int> send_counts(size, 0);
    std::vector<int> displs(size, 0);

    if (rank == 0) {
        if (strategy == PartitionStrategy::EQUAL_CHUNKS) {
            int base_chunk = static_cast<int>(total_valid_records / size);
            int remainder = static_cast<int>(total_valid_records % size);
            int offset = 0;
            for (int i = 0; i < size; ++i) {
                send_counts[i] = base_chunk + (i < remainder ? 1 : 0);
                displs[i] = offset;
                offset += send_counts[i];
            }
        } else {
            // Contiguous Time-Range Partitioning
            if (total_valid_records > 0) {
                int64_t t_min = load_res.records.front().timestamp_sec;
                int64_t t_max = load_res.records.back().timestamp_sec;
                int64_t t_span = (t_max - t_min + 1);
                int64_t t_step = std::max<int64_t>(1, t_span / size);

                size_t current_idx = 0;
                for (int i = 0; i < size; ++i) {
                    int64_t t_cutoff = (i == size - 1) ? (t_max + 1) : (t_min + (i + 1) * t_step);
                    size_t start_idx = current_idx;
                    while (current_idx < load_res.records.size() && load_res.records[current_idx].timestamp_sec < t_cutoff) {
                        current_idx++;
                    }
                    send_counts[i] = static_cast<int>(current_idx - start_idx);
                    displs[i] = static_cast<int>(start_idx);
                }
            }
        }
    }

    int local_count = 0;
    MPI_Scatter(send_counts.data(), 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<PowerReading> local_records(local_count);
    MPI_Scatterv(
        (rank == 0) ? load_res.records.data() : nullptr,
        send_counts.data(),
        displs.data(),
        mpi_reading_type,
        local_records.data(),
        local_count,
        mpi_reading_type,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end_dist = MPI_Wtime();
    res.timing.t_distribution = t_end_dist - t_start_dist;

    // 3. OpenMP Local Computation
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start_comp = MPI_Wtime();

    double local_sum = 0.0;
    double local_sum_sq = 0.0;
    double local_min = (local_count > 0) ? 1e9 : 0.0;
    double local_max = (local_count > 0) ? -1e9 : 0.0;
    double local_solar = 0.0;
    double local_wind = 0.0;
    double local_cost = 0.0;
    double local_voltage = 0.0;
    double local_pf = 0.0;
    int64_t local_overloads = 0;
    int64_t local_faults = 0;

    PeakCandidate local_peak = {0.0, 0, "", ""};

    std::vector<AggregationStats> local_hourly(24);
    std::map<std::string, AggregationStats> local_daily;

    #pragma omp parallel
    {
        double t_sum = 0.0;
        double t_sum_sq = 0.0;
        double t_min = 1e9;
        double t_max = -1e9;
        double t_solar = 0.0;
        double t_wind = 0.0;
        double t_cost = 0.0;
        double t_voltage = 0.0;
        double t_pf = 0.0;
        int64_t t_overloads = 0;
        int64_t t_faults = 0;

        PeakCandidate t_peak = {0.0, 0, "", ""};
        std::vector<AggregationStats> t_hourly(24);
        std::map<std::string, AggregationStats> t_daily;

        #pragma omp for nowait
        for (int i = 0; i < local_count; ++i) {
            const auto& r = local_records[i];
            double p = r.power_consumption;
            t_sum += p;
            t_sum_sq += (p * p);
            t_solar += r.solar_power;
            t_wind += r.wind_power;
            t_cost += (p * interval_hours * r.electricity_price);
            t_voltage += r.voltage;
            t_pf += r.power_factor;

            if (r.overload_condition == 1) t_overloads++;
            if (r.transformer_fault == 1) t_faults++;

            if (p < t_min) t_min = p;
            if (p > t_max) {
                t_max = p;
                t_peak.max_power = p;
                t_peak.timestamp_sec = r.timestamp_sec;
                strncpy(t_peak.date_str, r.date_str, sizeof(t_peak.date_str) - 1);
                strncpy(t_peak.time_str, r.time_str, sizeof(t_peak.time_str) - 1);
            }

            if (r.hour >= 0 && r.hour < 24) {
                auto& h = t_hourly[r.hour];
                h.count++;
                h.sum_power += p;
                h.sum_power_sq += (p * p);
                h.sum_solar += r.solar_power;
                h.sum_wind += r.wind_power;
                h.sum_cost += (p * interval_hours * r.electricity_price);
                if (r.overload_condition == 1) h.overload_count++;
                if (r.transformer_fault == 1) h.fault_count++;
                if (p < h.min_power) h.min_power = p;
                if (p > h.max_power) h.max_power = p;
            }

            std::string date_str(r.date_str);
            auto& d = t_daily[date_str];
            d.count++;
            d.sum_power += p;
            d.sum_power_sq += (p * p);
            d.sum_solar += r.solar_power;
            d.sum_wind += r.wind_power;
            d.sum_cost += (p * interval_hours * r.electricity_price);
            if (r.overload_condition == 1) d.overload_count++;
            if (r.transformer_fault == 1) d.fault_count++;
            if (p < d.min_power) d.min_power = p;
            if (p > d.max_power) d.max_power = p;
        }

        #pragma omp critical
        {
            local_sum += t_sum;
            local_sum_sq += t_sum_sq;
            local_solar += t_solar;
            local_wind += t_wind;
            local_cost += t_cost;
            local_voltage += t_voltage;
            local_pf += t_pf;
            local_overloads += t_overloads;
            local_faults += t_faults;

            if (t_min < local_min) local_min = t_min;
            if (t_max > local_max) {
                local_max = t_max;
                local_peak = t_peak;
            }

            for (int h = 0; h < 24; ++h) {
                if (t_hourly[h].count > 0) {
                    local_hourly[h].count += t_hourly[h].count;
                    local_hourly[h].sum_power += t_hourly[h].sum_power;
                    local_hourly[h].sum_power_sq += t_hourly[h].sum_power_sq;
                    local_hourly[h].sum_solar += t_hourly[h].sum_solar;
                    local_hourly[h].sum_wind += t_hourly[h].sum_wind;
                    local_hourly[h].sum_cost += t_hourly[h].sum_cost;
                    local_hourly[h].overload_count += t_hourly[h].overload_count;
                    local_hourly[h].fault_count += t_hourly[h].fault_count;
                    if (t_hourly[h].min_power < local_hourly[h].min_power) {
                        local_hourly[h].min_power = t_hourly[h].min_power;
                    }
                    if (t_hourly[h].max_power > local_hourly[h].max_power) {
                        local_hourly[h].max_power = t_hourly[h].max_power;
                    }
                }
            }

            for (const auto& pair : t_daily) {
                auto& dest = local_daily[pair.first];
                dest.count += pair.second.count;
                dest.sum_power += pair.second.sum_power;
                dest.sum_power_sq += pair.second.sum_power_sq;
                dest.sum_solar += pair.second.sum_solar;
                dest.sum_wind += pair.second.sum_wind;
                dest.sum_cost += pair.second.sum_cost;
                dest.overload_count += pair.second.overload_count;
                dest.fault_count += pair.second.fault_count;
                if (pair.second.min_power < dest.min_power) {
                    dest.min_power = pair.second.min_power;
                }
                if (pair.second.max_power > dest.max_power) {
                    dest.max_power = pair.second.max_power;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end_comp = MPI_Wtime();
    res.timing.t_computation = t_end_comp - t_start_comp;

    // 4. MPI Aggregation
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start_agg = MPI_Wtime();

    double global_sum = 0.0;
    double global_sum_sq = 0.0;
    double global_min = 0.0;
    double global_max = 0.0;
    double global_solar = 0.0;
    double global_wind = 0.0;
    double global_cost = 0.0;
    double global_voltage = 0.0;
    double global_pf = 0.0;
    int64_t global_overloads = 0;
    int64_t global_faults = 0;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_solar, &global_solar, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_wind, &global_wind, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cost, &global_cost, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_voltage, &global_voltage, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_pf, &global_pf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_overloads, &global_overloads, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_faults, &global_faults, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<PeakCandidate> gathered_peaks(size);
    MPI_Gather(&local_peak, 1, mpi_peak_type, gathered_peaks.data(), 1, mpi_peak_type, 0, MPI_COMM_WORLD);

    // Hourly reductions
    int64_t local_h_counts[24];
    double local_h_sums[24];
    double local_h_sums_sq[24];
    double local_h_mins[24];
    double local_h_maxs[24];
    double local_h_solar[24];
    double local_h_wind[24];
    int64_t local_h_overloads[24];
    int64_t local_h_faults[24];

    for (int h = 0; h < 24; ++h) {
        local_h_counts[h] = local_hourly[h].count;
        local_h_sums[h] = local_hourly[h].sum_power;
        local_h_sums_sq[h] = local_hourly[h].sum_power_sq;
        local_h_mins[h] = (local_hourly[h].count > 0) ? local_hourly[h].min_power : 1e9;
        local_h_maxs[h] = (local_hourly[h].count > 0) ? local_hourly[h].max_power : -1e9;
        local_h_solar[h] = local_hourly[h].sum_solar;
        local_h_wind[h] = local_hourly[h].sum_wind;
        local_h_overloads[h] = local_hourly[h].overload_count;
        local_h_faults[h] = local_hourly[h].fault_count;
    }

    int64_t global_h_counts[24];
    double global_h_sums[24];
    double global_h_sums_sq[24];
    double global_h_mins[24];
    double global_h_maxs[24];
    double global_h_solar[24];
    double global_h_wind[24];
    int64_t global_h_overloads[24];
    int64_t global_h_faults[24];

    MPI_Reduce(local_h_counts, global_h_counts, 24, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_h_sums, global_h_sums, 24, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_h_sums_sq, global_h_sums_sq, 24, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_h_mins, global_h_mins, 24, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_h_maxs, global_h_maxs, 24, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_h_solar, global_h_solar, 24, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_h_wind, global_h_wind, 24, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_h_overloads, global_h_overloads, 24, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_h_faults, global_h_faults, 24, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    // Daily Gatherv
    std::vector<DailyTransferEntry> local_daily_entries;
    local_daily_entries.reserve(local_daily.size());
    for (const auto& pair : local_daily) {
        DailyTransferEntry de;
        strncpy(de.date_str, pair.first.c_str(), sizeof(de.date_str) - 1);
        de.date_str[sizeof(de.date_str) - 1] = '\0';
        de.count = pair.second.count;
        de.sum_power = pair.second.sum_power;
        de.sum_power_sq = pair.second.sum_power_sq;
        de.min_power = pair.second.min_power;
        de.max_power = pair.second.max_power;
        de.sum_solar = pair.second.sum_solar;
        de.sum_wind = pair.second.sum_wind;
        de.sum_cost = pair.second.sum_cost;
        de.overload_count = pair.second.overload_count;
        de.fault_count = pair.second.fault_count;
        local_daily_entries.push_back(de);
    }

    int local_daily_count = static_cast<int>(local_daily_entries.size());
    std::vector<int> daily_counts(size, 0);
    std::vector<int> daily_displs(size, 0);

    MPI_Gather(&local_daily_count, 1, MPI_INT, daily_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_daily_entries = 0;
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            daily_displs[i] = total_daily_entries;
            total_daily_entries += daily_counts[i];
        }
    }

    std::vector<DailyTransferEntry> all_daily_entries(total_daily_entries);
    MPI_Gatherv(
        local_daily_entries.data(),
        local_daily_count,
        mpi_daily_type,
        all_daily_entries.data(),
        daily_counts.data(),
        daily_displs.data(),
        mpi_daily_type,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end_agg = MPI_Wtime();
    res.timing.t_aggregation = t_end_agg - t_start_agg;

    // 5. Global Threshold Broadcast
    double global_mean = 0.0;
    double global_stddev = 0.0;
    double global_threshold = 0.0;

    if (rank == 0) {
        if (total_valid_records > 0) {
            global_mean = global_sum / total_valid_records;
            double variance = (global_sum_sq / total_valid_records) - (global_mean * global_mean);
            global_stddev = (variance > 0.0) ? std::sqrt(variance) : 0.0;
            global_threshold = global_mean + (anomaly_k_factor * global_stddev);
        }
    }

    MPI_Bcast(&global_mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 6. Parallel Anomaly Detection
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start_anom = MPI_Wtime();

    std::vector<AnomalyTransferEntry> local_anomalies;
    #pragma omp parallel
    {
        std::vector<AnomalyTransferEntry> t_anomalies;
        #pragma omp for nowait
        for (int i = 0; i < local_count; ++i) {
            const auto& r = local_records[i];
            bool is_statistical_spike = (r.power_consumption > global_threshold);
            bool is_grid_overload = (r.overload_condition == 1);
            bool is_transformer_fault = (r.transformer_fault == 1);

            if (is_statistical_spike || is_grid_overload || is_transformer_fault) {
                AnomalyTransferEntry ae;
                snprintf(ae.timestamp_str, sizeof(ae.timestamp_str), "%s %s", r.date_str, r.time_str);
                ae.timestamp_sec = r.timestamp_sec;
                ae.active_power = r.power_consumption;
                ae.voltage = r.voltage;
                ae.z_score = (global_stddev > 0.0) ? ((r.power_consumption - global_mean) / global_stddev) : 0.0;
                ae.is_overload = r.overload_condition;
                ae.is_fault = r.transformer_fault;

                if (is_transformer_fault) {
                    strncpy(ae.alert_category, "TRANSFORMER_FAULT", sizeof(ae.alert_category) - 1);
                } else if (is_grid_overload && is_statistical_spike) {
                    strncpy(ae.alert_category, "OVERLOAD_AND_SPIKE", sizeof(ae.alert_category) - 1);
                } else if (is_grid_overload) {
                    strncpy(ae.alert_category, "GRID_OVERLOAD", sizeof(ae.alert_category) - 1);
                } else {
                    strncpy(ae.alert_category, "STATISTICAL_SPIKE", sizeof(ae.alert_category) - 1);
                }
                ae.alert_category[sizeof(ae.alert_category) - 1] = '\0';
                t_anomalies.push_back(ae);
            }
        }
        #pragma omp critical
        {
            local_anomalies.insert(local_anomalies.end(), t_anomalies.begin(), t_anomalies.end());
        }
    }

    int local_anom_count = static_cast<int>(local_anomalies.size());
    std::vector<int> anom_counts(size, 0);
    std::vector<int> anom_displs(size, 0);

    MPI_Gather(&local_anom_count, 1, MPI_INT, anom_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_anomalies = 0;
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            anom_displs[i] = total_anomalies;
            total_anomalies += anom_counts[i];
        }
    }

    std::vector<AnomalyTransferEntry> all_anomalies(total_anomalies);
    MPI_Gatherv(
        local_anomalies.data(),
        local_anom_count,
        mpi_anomaly_type,
        all_anomalies.data(),
        anom_counts.data(),
        anom_displs.data(),
        mpi_anomaly_type,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end_anom = MPI_Wtime();
    res.timing.t_anomaly_detect = t_end_anom - t_start_anom;

    double t_end_total = MPI_Wtime();
    res.timing.t_end_to_end = t_end_total - t_start_total;
    res.timing.t_parallel_total = res.timing.t_distribution + res.timing.t_computation + res.timing.t_aggregation + res.timing.t_anomaly_detect;

    // 7. Assemble Final Global Results on Rank 0
    if (rank == 0) {
        res.summary.total_records_read = total_lines_read;
        res.summary.valid_records = total_valid_records;
        res.summary.rejected_records = total_rejected;
        res.summary.sampling_interval_minutes = detected_interval_minutes;
        res.summary.sampling_interval_hours = interval_hours;
        res.summary.anomaly_k_factor = anomaly_k_factor;

        res.summary.mean_active_power = global_mean;
        res.summary.stddev_active_power = global_stddev;
        res.summary.variance_active_power = global_stddev * global_stddev;
        res.summary.min_active_power = global_min;
        res.summary.max_active_power = global_max;
        res.summary.total_energy_kwh = global_sum * interval_hours;
        res.summary.total_solar_energy_kwh = global_solar * interval_hours;
        res.summary.total_wind_energy_kwh = global_wind * interval_hours;
        res.summary.total_renewable_energy_kwh = res.summary.total_solar_energy_kwh + res.summary.total_wind_energy_kwh;
        res.summary.total_cost_usd = global_cost;
        res.summary.avg_voltage = (total_valid_records > 0) ? (global_voltage / total_valid_records) : 0.0;
        res.summary.avg_power_factor = (total_valid_records > 0) ? (global_pf / total_valid_records) : 0.0;
        res.summary.total_overload_intervals = global_overloads;
        res.summary.total_transformer_faults = global_faults;
        res.summary.anomaly_threshold = global_threshold;
        res.summary.anomaly_count = total_anomalies;

        // Resolve global peak candidate
        PeakCandidate true_peak = { -1e9, 0, "", "" };
        for (const auto& cand : gathered_peaks) {
            if (cand.max_power > true_peak.max_power) {
                true_peak = cand;
            }
        }
        res.summary.peak_demand_power = true_peak.max_power;
        res.summary.peak_timestamp_str = std::string(true_peak.date_str) + " " + std::string(true_peak.time_str);
        res.summary.peak_timestamp_sec = true_peak.timestamp_sec;

        // Assemble hourly stats
        res.hourly.resize(24);
        for (int h = 0; h < 24; ++h) {
            res.hourly[h].count = global_h_counts[h];
            res.hourly[h].sum_power = global_h_sums[h];
            res.hourly[h].sum_power_sq = global_h_sums_sq[h];
            res.hourly[h].min_power = (global_h_counts[h] > 0) ? global_h_mins[h] : 0.0;
            res.hourly[h].max_power = (global_h_counts[h] > 0) ? global_h_maxs[h] : 0.0;
            res.hourly[h].sum_solar = global_h_solar[h];
            res.hourly[h].sum_wind = global_h_wind[h];
            res.hourly[h].overload_count = global_h_overloads[h];
            res.hourly[h].fault_count = global_h_faults[h];
        }

        // Peak hour
        double highest_hour_avg = -1.0;
        double highest_hour_max = -1.0;
        for (int h = 0; h < 24; ++h) {
            if (res.hourly[h].count > 0) {
                double avg = res.hourly[h].mean();
                if (avg > highest_hour_avg) {
                    highest_hour_avg = avg;
                    res.summary.peak_hour_avg = h;
                    res.summary.peak_hour_avg_power = avg;
                }
                if (res.hourly[h].max_power > highest_hour_max) {
                    highest_hour_max = res.hourly[h].max_power;
                    res.summary.peak_hour_max = h;
                    res.summary.peak_hour_max_power = res.hourly[h].max_power;
                }
            }
        }

        // Assemble daily stats
        for (const auto& entry : all_daily_entries) {
            std::string date_key(entry.date_str);
            auto& d = res.daily[date_key];
            d.count += entry.count;
            d.sum_power += entry.sum_power;
            d.sum_power_sq += entry.sum_power_sq;
            d.sum_solar += entry.sum_solar;
            d.sum_wind += entry.sum_wind;
            d.sum_cost += entry.sum_cost;
            d.overload_count += entry.overload_count;
            d.fault_count += entry.fault_count;
            if (entry.min_power < d.min_power) d.min_power = entry.min_power;
            if (entry.max_power > d.max_power) d.max_power = entry.max_power;
        }

        // Peak day
        double highest_day_avg = -1.0;
        double highest_day_max = -1.0;
        for (const auto& pair : res.daily) {
            if (pair.second.count > 0) {
                double avg = pair.second.mean();
                if (avg > highest_day_avg) {
                    highest_day_avg = avg;
                    res.summary.peak_day_avg = pair.first;
                    res.summary.peak_day_avg_power = avg;
                }
                if (pair.second.max_power > highest_day_max) {
                    highest_day_max = pair.second.max_power;
                    res.summary.peak_day_max = pair.first;
                    res.summary.peak_day_max_power = pair.second.max_power;
                }
            }
        }

        // Assemble anomalies
        res.anomalies.reserve(total_anomalies);
        for (const auto& ae : all_anomalies) {
            AnomalyRecord ar;
            ar.timestamp_str = ae.timestamp_str;
            ar.timestamp_sec = ae.timestamp_sec;
            ar.active_power = ae.active_power;
            ar.voltage = ae.voltage;
            ar.z_score = ae.z_score;
            ar.is_overload = ae.is_overload;
            ar.is_fault = ae.is_fault;
            ar.alert_category = ae.alert_category;
            res.anomalies.push_back(ar);
        }

        std::sort(res.anomalies.begin(), res.anomalies.end(), [](const AnomalyRecord& a, const AnomalyRecord& b) {
            return a.timestamp_sec < b.timestamp_sec;
        });

        if (!output_dir.empty()) {
            try {
                fs::create_directories(output_dir);
                Statistics::export_summary_csv(output_dir + "/summary.csv", res.summary, res.timing);
                Statistics::export_hourly_csv(output_dir + "/hourly_stats.csv", res.hourly, interval_hours);
                Statistics::export_daily_csv(output_dir + "/daily_stats.csv", res.daily, interval_hours);
                Statistics::export_anomalies_csv(output_dir + "/anomalies.csv", res.anomalies);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to write parallel output files: " << e.what() << "\n";
            }
        }

        res.success = true;
    }

    MPI_Type_free(&mpi_reading_type);
    MPI_Type_free(&mpi_daily_type);
    MPI_Type_free(&mpi_anomaly_type);
    MPI_Type_free(&mpi_peak_type);

    if (finalized_mpi_locally) {
        MPI_Finalize();
    }

    return res;
}
