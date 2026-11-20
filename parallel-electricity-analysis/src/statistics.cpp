#include "statistics.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

std::vector<AggregationStats> Statistics::compute_hourly(const std::vector<PowerReading>& readings, double /*interval_hours*/) {
    std::vector<AggregationStats> hourly(24);
    for (const auto& r : readings) {
        if (r.hour >= 0 && r.hour < 24) {
            auto& h = hourly[r.hour];
            h.count++;
            h.sum_power += r.power_consumption;
            h.sum_power_sq += (r.power_consumption * r.power_consumption);
            h.sum_solar += r.solar_power;
            h.sum_wind += r.wind_power;
            h.sum_cost += (r.power_consumption * 0.25 * r.electricity_price);
            if (r.overload_condition == 1) h.overload_count++;
            if (r.transformer_fault == 1) h.fault_count++;
            if (r.power_consumption < h.min_power) h.min_power = r.power_consumption;
            if (r.power_consumption > h.max_power) h.max_power = r.power_consumption;
        }
    }
    for (int h = 0; h < 24; ++h) {
        if (hourly[h].count == 0) {
            hourly[h].min_power = 0.0;
            hourly[h].max_power = 0.0;
        }
    }
    return hourly;
}

std::map<std::string, AggregationStats> Statistics::compute_daily(const std::vector<PowerReading>& readings, double /*interval_hours*/) {
    std::map<std::string, AggregationStats> daily;
    for (const auto& r : readings) {
        std::string date_key(r.date_str);
        auto& stats = daily[date_key];
        stats.count++;
        stats.sum_power += r.power_consumption;
        stats.sum_power_sq += (r.power_consumption * r.power_consumption);
        stats.sum_solar += r.solar_power;
        stats.sum_wind += r.wind_power;
        stats.sum_cost += (r.power_consumption * 0.25 * r.electricity_price);
        if (r.overload_condition == 1) stats.overload_count++;
        if (r.transformer_fault == 1) stats.fault_count++;
        if (r.power_consumption < stats.min_power) stats.min_power = r.power_consumption;
        if (r.power_consumption > stats.max_power) stats.max_power = r.power_consumption;
    }
    for (auto& pair : daily) {
        if (pair.second.count == 0) {
            pair.second.min_power = 0.0;
            pair.second.max_power = 0.0;
        }
    }
    return daily;
}

GlobalSummary Statistics::compute_global_summary(
    const std::vector<PowerReading>& readings,
    int64_t total_lines_read,
    int64_t rejected_count,
    double interval_minutes,
    const std::vector<AggregationStats>& hourly,
    const std::map<std::string, AggregationStats>& daily,
    double anomaly_k_factor
) {
    GlobalSummary summary;
    summary.total_records_read = total_lines_read;
    summary.valid_records = static_cast<int64_t>(readings.size());
    summary.rejected_records = rejected_count;
    summary.sampling_interval_minutes = interval_minutes;
    summary.sampling_interval_hours = interval_minutes / 60.0;
    summary.anomaly_k_factor = anomaly_k_factor;

    if (readings.empty()) {
        return summary;
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    double min_val = 1e9;
    double max_val = -1e9;
    size_t peak_idx = 0;

    double sum_solar = 0.0;
    double sum_wind = 0.0;
    double sum_cost = 0.0;
    double sum_voltage = 0.0;
    double sum_pf = 0.0;
    int64_t overloads = 0;
    int64_t faults = 0;

    for (size_t i = 0; i < readings.size(); ++i) {
        const auto& r = readings[i];
        double p = r.power_consumption;
        sum += p;
        sum_sq += (p * p);
        if (p < min_val) min_val = p;
        if (p > max_val) {
            max_val = p;
            peak_idx = i;
        }

        sum_solar += r.solar_power;
        sum_wind += r.wind_power;
        sum_cost += (p * summary.sampling_interval_hours * r.electricity_price);
        sum_voltage += r.voltage;
        sum_pf += r.power_factor;
        if (r.overload_condition == 1) overloads++;
        if (r.transformer_fault == 1) faults++;
    }

    summary.min_active_power = min_val;
    summary.max_active_power = max_val;
    summary.mean_active_power = sum / summary.valid_records;
    double variance = (sum_sq / summary.valid_records) - (summary.mean_active_power * summary.mean_active_power);
    summary.variance_active_power = (variance < 0.0) ? 0.0 : variance;
    summary.stddev_active_power = std::sqrt(summary.variance_active_power);

    summary.total_energy_kwh = sum * summary.sampling_interval_hours;
    summary.total_solar_energy_kwh = sum_solar * summary.sampling_interval_hours;
    summary.total_wind_energy_kwh = sum_wind * summary.sampling_interval_hours;
    summary.total_renewable_energy_kwh = summary.total_solar_energy_kwh + summary.total_wind_energy_kwh;
    summary.total_cost_usd = sum_cost;
    summary.avg_voltage = sum_voltage / summary.valid_records;
    summary.avg_power_factor = sum_pf / summary.valid_records;
    summary.total_overload_intervals = overloads;
    summary.total_transformer_faults = faults;

    // Peak demand info
    summary.peak_demand_power = max_val;
    summary.peak_timestamp_str = std::string(readings[peak_idx].date_str) + " " + std::string(readings[peak_idx].time_str);
    summary.peak_timestamp_sec = readings[peak_idx].timestamp_sec;

    // Peak hour
    double highest_hour_avg = -1.0;
    double highest_hour_max = -1.0;
    for (int h = 0; h < 24; ++h) {
        if (hourly[h].count > 0) {
            double avg = hourly[h].mean();
            if (avg > highest_hour_avg) {
                highest_hour_avg = avg;
                summary.peak_hour_avg = h;
                summary.peak_hour_avg_power = avg;
            }
            if (hourly[h].max_power > highest_hour_max) {
                highest_hour_max = hourly[h].max_power;
                summary.peak_hour_max = h;
                summary.peak_hour_max_power = hourly[h].max_power;
            }
        }
    }

    // Peak day
    double highest_day_avg = -1.0;
    double highest_day_max = -1.0;
    for (const auto& pair : daily) {
        if (pair.second.count > 0) {
            double avg = pair.second.mean();
            if (avg > highest_day_avg) {
                highest_day_avg = avg;
                summary.peak_day_avg = pair.first;
                summary.peak_day_avg_power = avg;
            }
            if (pair.second.max_power > highest_day_max) {
                highest_day_max = pair.second.max_power;
                summary.peak_day_max = pair.first;
                summary.peak_day_max_power = pair.second.max_power;
            }
        }
    }

    // Anomaly threshold: Mean + k * StdDev
    summary.anomaly_threshold = summary.mean_active_power + (summary.anomaly_k_factor * summary.stddev_active_power);

    return summary;
}

std::vector<AnomalyRecord> Statistics::detect_anomalies(
    const std::vector<PowerReading>& readings,
    double mean,
    double stddev,
    double threshold
) {
    std::vector<AnomalyRecord> anomalies;
    for (const auto& r : readings) {
        bool is_statistical_spike = (r.power_consumption > threshold);
        bool is_grid_overload = (r.overload_condition == 1);
        bool is_transformer_fault = (r.transformer_fault == 1);

        if (is_statistical_spike || is_grid_overload || is_transformer_fault) {
            AnomalyRecord ar;
            ar.timestamp_str = std::string(r.date_str) + " " + std::string(r.time_str);
            ar.timestamp_sec = r.timestamp_sec;
            ar.active_power = r.power_consumption;
            ar.z_score = (stddev > 0.0) ? ((r.power_consumption - mean) / stddev) : 0.0;
            ar.is_overload = r.overload_condition;
            ar.is_fault = r.transformer_fault;
            ar.voltage = r.voltage;

            if (is_transformer_fault) {
                ar.alert_category = "TRANSFORMER_FAULT";
            } else if (is_grid_overload && is_statistical_spike) {
                ar.alert_category = "OVERLOAD_AND_SPIKE";
            } else if (is_grid_overload) {
                ar.alert_category = "GRID_OVERLOAD";
            } else {
                ar.alert_category = "STATISTICAL_SPIKE";
            }

            anomalies.push_back(ar);
        }
    }
    return anomalies;
}

bool Statistics::export_summary_csv(const std::string& filepath, const GlobalSummary& s, const ExecutionTiming& t) {
    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << "Metric,Value,Unit,Description\n";
    out << "Dataset_Source,Kaggle Smart Grid Real-Time Load Monitoring,source,Kaggle ziya07/smart-grid-real-time-load-monitoring-dataset\n";
    out << "Total_Records_Read," << s.total_records_read << ",count,Total telemetry readings read from CSV\n";
    out << "Valid_Records," << s.valid_records << ",count,Clean valid telemetry readings processed\n";
    out << "Rejected_Records," << s.rejected_records << ",count,Malformed or corrupt records filtered out\n";
    out << "Sampling_Interval_Minutes," << s.sampling_interval_minutes << ",minutes,Time resolution per measurement\n";
    out << std::fixed << std::setprecision(4);
    out << "Mean_Active_Power," << s.mean_active_power << ",kW,Average grid active load demand\n";
    out << "StdDev_Active_Power," << s.stddev_active_power << ",kW,Standard deviation of load demand\n";
    out << "Min_Active_Power," << s.min_active_power << ",kW,Minimum recorded active power\n";
    out << "Max_Active_Power," << s.max_active_power << ",kW,Peak recorded active power spike\n";
    out << "Peak_Timestamp," << s.peak_timestamp_str << ",timestamp,Timestamp of peak load spike\n";
    out << "Total_Energy_Consumption," << s.total_energy_kwh << ",kWh,Total cumulative electrical energy consumed\n";
    out << "Total_Solar_Energy," << s.total_solar_energy_kwh << ",kWh,Renewable energy generated from solar\n";
    out << "Total_Wind_Energy," << s.total_wind_energy_kwh << ",kWh,Renewable energy generated from wind\n";
    out << "Total_Renewable_Energy," << s.total_renewable_energy_kwh << ",kWh,Sum of solar and wind generation\n";
    out << "Total_Energy_Cost," << s.total_cost_usd << ",USD,Estimated dynamic tariff electricity cost\n";
    out << "Average_Grid_Voltage," << s.avg_voltage << ",V,Mean line voltage\n";
    out << "Average_Power_Factor," << s.avg_power_factor << ",ratio,Mean grid power factor\n";
    out << "Total_Overload_Intervals," << s.total_overload_intervals << ",count,Count of 15-min intervals with overload alert active\n";
    out << "Total_Transformer_Faults," << s.total_transformer_faults << ",count,Count of 15-min intervals with transformer fault alert\n";
    out << "Peak_Hour_By_Average," << s.peak_hour_avg << ",hour_of_day,Diurnal hour (0-23) with highest average load\n";
    out << "Peak_Hour_Average_Power," << s.peak_hour_avg_power << ",kW,Average demand during peak hour\n";
    out << "Peak_Hour_By_Max," << s.peak_hour_max << ",hour_of_day,Diurnal hour containing highest spike\n";
    out << "Peak_Day_By_Average," << s.peak_day_avg << ",date,Calendar day with highest average power\n";
    out << "Peak_Day_Average_Power," << s.peak_day_avg_power << ",kW,Average load during peak day\n";
    out << "Peak_Day_By_Max," << s.peak_day_max << ",date,Calendar day containing single largest load spike\n";
    out << "Anomaly_K_Factor," << s.anomaly_k_factor << ",multiplier,Multiplier k for Mean + k*StdDev\n";
    out << "Anomaly_Threshold," << s.anomaly_threshold << ",kW,Statistical threshold for spike detection\n";
    out << "Total_Flagged_Anomalies," << s.anomaly_count << ",count,Total flagged anomalies (spikes/overloads/faults)\n";
    out << "MPI_Processes," << t.mpi_ranks << ",ranks,Number of MPI processes\n";
    out << "OpenMP_Threads," << t.omp_threads << ",threads,Number of OpenMP threads per process\n";
    out << "Partitioning_Strategy," << t.partitioning_strategy << ",strategy,Partitioning strategy used\n";
    out << "Time_Data_Loading," << t.t_load_preproc << ",seconds,File reading and preprocessing time\n";
    out << "Time_Data_Distribution," << t.t_distribution << ",seconds,MPI scatter and partition time\n";
    out << "Time_Computation," << t.t_computation << ",seconds,Parallel local statistical computation time\n";
    out << "Time_MPI_Aggregation," << t.t_aggregation << ",seconds,MPI reduction and gathering time\n";
    out << "Time_Anomaly_Detection," << t.t_anomaly_detect << ",seconds,Anomaly filtering and collection time\n";
    out << "Time_Parallel_Phase," << t.t_parallel_total << ",seconds,Sum of parallel computation and communication\n";
    out << "Time_End_To_End," << t.t_end_to_end << ",seconds,Total wall-clock runtime\n";

    return true;
}

bool Statistics::export_hourly_csv(const std::string& filepath, const std::vector<AggregationStats>& hourly, double interval_hours) {
    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << "Hour,Count,Mean_Power_kW,StdDev_kW,Min_Power_kW,Max_Power_kW,Total_Energy_kWh,Solar_kWh,Wind_kWh,Renewable_kWh,Overloads,Faults\n";
    out << std::fixed << std::setprecision(4);
    for (int h = 0; h < 24; ++h) {
        double e_kwh = hourly[h].total_energy_kwh(interval_hours);
        double s_kwh = hourly[h].sum_solar * interval_hours;
        double w_kwh = hourly[h].sum_wind * interval_hours;
        out << h << ","
            << hourly[h].count << ","
            << hourly[h].mean() << ","
            << hourly[h].std_dev() << ","
            << hourly[h].min_power << ","
            << hourly[h].max_power << ","
            << e_kwh << ","
            << s_kwh << ","
            << w_kwh << ","
            << (s_kwh + w_kwh) << ","
            << hourly[h].overload_count << ","
            << hourly[h].fault_count << "\n";
    }
    return true;
}

bool Statistics::export_daily_csv(const std::string& filepath, const std::map<std::string, AggregationStats>& daily, double interval_hours) {
    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << "Date,Count,Mean_Power_kW,StdDev_kW,Min_Power_kW,Max_Power_kW,Total_Energy_kWh,Renewable_kWh,Cost_USD,Overloads,Faults\n";
    out << std::fixed << std::setprecision(4);
    for (const auto& pair : daily) {
        double e_kwh = pair.second.total_energy_kwh(interval_hours);
        double ren_kwh = pair.second.total_renewable_energy_kwh(interval_hours);
        out << pair.first << ","
            << pair.second.count << ","
            << pair.second.mean() << ","
            << pair.second.std_dev() << ","
            << pair.second.min_power << ","
            << pair.second.max_power << ","
            << e_kwh << ","
            << ren_kwh << ","
            << pair.second.sum_cost << ","
            << pair.second.overload_count << ","
            << pair.second.fault_count << "\n";
    }
    return true;
}

bool Statistics::export_anomalies_csv(const std::string& filepath, const std::vector<AnomalyRecord>& anomalies) {
    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << "Timestamp,Epoch_Seconds,Active_Power_kW,Voltage_V,Z_Score,Is_Overload,Is_Fault,Alert_Category\n";
    out << std::fixed << std::setprecision(4);
    for (const auto& a : anomalies) {
        out << a.timestamp_str << ","
            << a.timestamp_sec << ","
            << a.active_power << ","
            << a.voltage << ","
            << a.z_score << ","
            << a.is_overload << ","
            << a.is_fault << ","
            << a.alert_category << "\n";
    }
    return true;
}

bool Statistics::export_benchmark_csv(const std::string& filepath, const std::vector<ExecutionTiming>& bench) {
    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << "Config_Name,MPI_Ranks,OMP_Threads,Total_Units,Strategy,T_Load_s,T_Dist_s,T_Comp_s,T_Comm_s,T_Parallel_s,T_EndToEnd_s,Speedup_E2E,Efficiency_E2E,Speedup_Comp,Efficiency_Comp\n";
    out << std::fixed << std::setprecision(6);

    double t_seq_e2e = 0.0;
    double t_seq_comp = 0.0;
    if (!bench.empty()) {
        t_seq_e2e = bench[0].t_end_to_end;
        t_seq_comp = bench[0].t_computation + bench[0].t_anomaly_detect;
    }

    for (const auto& b : bench) {
        int P = b.mpi_ranks * b.omp_threads;
        double s_e2e = (b.t_end_to_end > 0.0 && t_seq_e2e > 0.0) ? (t_seq_e2e / b.t_end_to_end) : 1.0;
        double e_e2e = (P > 0) ? (s_e2e / P) : 1.0;

        double t_par_comp = b.t_computation + b.t_anomaly_detect;
        double s_comp = (t_par_comp > 0.0 && t_seq_comp > 0.0) ? (t_seq_comp / t_par_comp) : 1.0;
        double e_comp = (P > 0) ? (s_comp / P) : 1.0;

        std::string config_name = std::to_string(b.mpi_ranks) + "P x " + std::to_string(b.omp_threads) + "T (" + b.partitioning_strategy + ")";
        out << config_name << ","
            << b.mpi_ranks << ","
            << b.omp_threads << ","
            << P << ","
            << b.partitioning_strategy << ","
            << b.t_load_preproc << ","
            << b.t_distribution << ","
            << b.t_computation << ","
            << b.t_aggregation << ","
            << b.t_parallel_total << ","
            << b.t_end_to_end << ","
            << s_e2e << ","
            << e_e2e << ","
            << s_comp << ","
            << e_comp << "\n";
    }
    return true;
}

void Statistics::print_summary_report(const GlobalSummary& s, const ExecutionTiming& t) {
    std::cout << "\n================================================================================\n";
    std::cout << "     SMART GRID REAL-TIME LOAD MONITORING & PEAK LOAD ANALYSIS (MPI+OpenMP)     \n";
    std::cout << "================================================================================\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << " [Dataset Ingestion & Telemetry]\n";
    std::cout << "   - Dataset Name           : Kaggle Smart Grid Real-Time Load Monitoring Dataset\n";
    std::cout << "   - Total Records Read     : " << s.total_records_read << "\n";
    std::cout << "   - Valid Records Processed: " << s.valid_records << "\n";
    std::cout << "   - Rejected / Malformed   : " << s.rejected_records << " ("
              << (s.total_records_read > 0 ? (100.0 * s.rejected_records / s.total_records_read) : 0.0) << "%)\n";
    std::cout << "   - Sampling Interval      : " << s.sampling_interval_minutes << " minutes (dt = " << s.sampling_interval_hours << " hr)\n";

    std::cout << "\n [Power & Energy Metrics]\n";
    std::cout << "   - Mean Active Power Load : " << s.mean_active_power << " kW\n";
    std::cout << "   - Standard Deviation     : " << s.stddev_active_power << " kW\n";
    std::cout << "   - Minimum Observed Load  : " << s.min_active_power << " kW\n";
    std::cout << "   - Peak Demand Spike      : " << s.peak_demand_power << " kW\n";
    std::cout << "   - Peak Spike Timestamp   : " << s.peak_timestamp_str << "\n";
    std::cout << "   - Total Energy Consumed  : " << s.total_energy_kwh << " kWh\n";
    std::cout << "   - Solar Generation       : " << s.total_solar_energy_kwh << " kWh\n";
    std::cout << "   - Wind Generation        : " << s.total_wind_energy_kwh << " kWh\n";
    std::cout << "   - Total Renewable Energy : " << s.total_renewable_energy_kwh << " kWh\n";
    std::cout << "   - Total Electricity Cost : $" << s.total_cost_usd << " USD\n";

    std::cout << "\n [Grid Stability & Reliability]\n";
    std::cout << "   - Average Line Voltage   : " << s.avg_voltage << " V (Nominal: 230V)\n";
    std::cout << "   - Average Power Factor   : " << s.avg_power_factor << "\n";
    std::cout << "   - Overload Alert Count   : " << s.total_overload_intervals << " intervals\n";
    std::cout << "   - Transformer Fault Count: " << s.total_transformer_faults << " intervals\n";

    std::cout << "\n [Peak Load Identification]\n";
    std::cout << "   - Peak Diurnal Hour (Avg): " << s.peak_hour_avg << ":00 (" << s.peak_hour_avg_power << " kW avg)\n";
    std::cout << "   - Peak Diurnal Hour (Max): " << s.peak_hour_max << ":00 (" << s.peak_hour_max_power << " kW max)\n";
    std::cout << "   - Peak Calendar Day (Avg): " << s.peak_day_avg << " (" << s.peak_day_avg_power << " kW avg)\n";
    std::cout << "   - Peak Calendar Day (Max): " << s.peak_day_max << " (" << s.peak_day_max_power << " kW max spike)\n";

    std::cout << "\n [Anomaly & Alert Detection]\n";
    std::cout << "   - Spike Detection Rule   : Reading > Mean + " << s.anomaly_k_factor << " * StdDev\n";
    std::cout << "   - Statistical Threshold  : " << s.anomaly_threshold << " kW\n";
    std::cout << "   - Total Flagged Events   : " << s.anomaly_count << " (Includes statistical spikes, overloads & faults)\n";

    std::cout << "\n [High Performance Computing Execution]\n";
    std::cout << "   - Parallel Hardware      : " << t.mpi_ranks << " MPI Process(es) x "
              << t.omp_threads << " OpenMP Thread(s) = " << (t.mpi_ranks * t.omp_threads) << " Core(s)\n";
    std::cout << "   - Partitioning Strategy  : " << t.partitioning_strategy << "\n";
    std::cout << "   - Data Ingestion Time    : " << t.t_load_preproc << " s\n";
    std::cout << "   - MPI Data Scatter Time  : " << t.t_distribution << " s\n";
    std::cout << "   - OpenMP Computation    : " << t.t_computation << " s\n";
    std::cout << "   - MPI Aggregation/Reduce : " << t.t_aggregation << " s\n";
    std::cout << "   - Anomaly Detection Time : " << t.t_anomaly_detect << " s\n";
    std::cout << "   - Parallel Phase Runtime : " << t.t_parallel_total << " s\n";
    std::cout << "   - End-to-End Wall Time   : " << t.t_end_to_end << " s\n";
    std::cout << "================================================================================\n\n";
}
