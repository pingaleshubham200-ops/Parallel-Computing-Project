#include "dataset_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <random>
#include <iomanip>
#include <algorithm>

int64_t DatasetLoader::to_epoch_seconds(int year, int month, int day, int hour, int minute, int second) {
    // Howard Hinnant algorithm for civil calendar to days since 1970-01-01
    year -= (month <= 2) ? 1 : 0;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(year - era * 400);
    const unsigned doy = (153 * (month > 2 ? month - 3 : month + 9) + 2) / 5 + day - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const int64_t days = era * 146097 + static_cast<int64_t>(doe) - 719468;
    return days * 86400LL + static_cast<int64_t>(hour) * 3600LL + static_cast<int64_t>(minute) * 60LL + second;
}

static std::vector<std::string> split_line(const std::string& line, char delim) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(line);
    while (std::getline(tokenStream, token, delim)) {
        // Strip trailing CR if any
        if (!token.empty() && token.back() == '\r') {
            token.pop_back();
        }
        tokens.push_back(token);
    }
    return tokens;
}

DatasetLoader::LoadResult DatasetLoader::load_from_file(const std::string& filepath, int64_t max_records) {
    LoadResult result;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        result.success = false;
        result.error_message = "Failed to open file: " + filepath;
        return result;
    }

    std::string header_line;
    if (!std::getline(file, header_line)) {
        result.success = false;
        result.error_message = "File is empty: " + filepath;
        return result;
    }

    // Determine delimiter
    char delimiter = ',';
    if (header_line.find(';') != std::string::npos && header_line.find(',') == std::string::npos) {
        delimiter = ';';
    }

    auto headers = split_line(header_line, delimiter);

    // Map column indices
    int idx_timestamp = -1;
    int idx_date = -1;
    int idx_time = -1;
    int idx_power = -1;
    int idx_voltage = -1;
    int idx_current = -1;
    int idx_reactive = -1;
    int idx_power_factor = -1;
    int idx_solar = -1;
    int idx_wind = -1;
    int idx_grid_supply = -1;
    int idx_volt_fluct = -1;
    int idx_overload = -1;
    int idx_fault = -1;
    int idx_temp = -1;
    int idx_humidity = -1;
    int idx_price = -1;
    int idx_pred_load = -1;

    for (size_t i = 0; i < headers.size(); ++i) {
        std::string h = headers[i];
        // Lowercase for comparison
        std::string h_lower = h;
        std::transform(h_lower.begin(), h_lower.end(), h_lower.begin(), ::tolower);

        if (h_lower == "timestamp") idx_timestamp = static_cast<int>(i);
        else if (h_lower == "date") idx_date = static_cast<int>(i);
        else if (h_lower == "time") idx_time = static_cast<int>(i);
        else if (h_lower.find("power consumption") != std::string::npos || h_lower.find("global_active_power") != std::string::npos) {
            idx_power = static_cast<int>(i);
        } else if (h_lower.find("voltage (v)") != std::string::npos || h_lower == "voltage") {
            idx_voltage = static_cast<int>(i);
        } else if (h_lower.find("current (a)") != std::string::npos || h_lower.find("global_intensity") != std::string::npos) {
            idx_current = static_cast<int>(i);
        } else if (h_lower.find("reactive power") != std::string::npos || h_lower.find("global_reactive_power") != std::string::npos) {
            idx_reactive = static_cast<int>(i);
        } else if (h_lower.find("power factor") != std::string::npos) {
            idx_power_factor = static_cast<int>(i);
        } else if (h_lower.find("solar power") != std::string::npos) {
            idx_solar = static_cast<int>(i);
        } else if (h_lower.find("wind power") != std::string::npos) {
            idx_wind = static_cast<int>(i);
        } else if (h_lower.find("grid supply") != std::string::npos) {
            idx_grid_supply = static_cast<int>(i);
        } else if (h_lower.find("voltage fluctuation") != std::string::npos) {
            idx_volt_fluct = static_cast<int>(i);
        } else if (h_lower.find("overload condition") != std::string::npos) {
            idx_overload = static_cast<int>(i);
        } else if (h_lower.find("transformer fault") != std::string::npos) {
            idx_fault = static_cast<int>(i);
        } else if (h_lower.find("temperature") != std::string::npos) {
            idx_temp = static_cast<int>(i);
        } else if (h_lower.find("humidity") != std::string::npos) {
            idx_humidity = static_cast<int>(i);
        } else if (h_lower.find("electricity price") != std::string::npos) {
            idx_price = static_cast<int>(i);
        } else if (h_lower.find("predicted load") != std::string::npos) {
            idx_pred_load = static_cast<int>(i);
        }
    }

    // Default power index if not found
    if (idx_power == -1) {
        if (headers.size() >= 4) idx_power = 3;
        else if (headers.size() >= 3) idx_power = 2;
        else idx_power = 0;
    }

    result.records.reserve(60000);
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        result.total_lines_read++;

        if (max_records > 0 && result.valid_records >= max_records) {
            break;
        }

        auto tokens = split_line(line, delimiter);
        if (tokens.size() <= static_cast<size_t>(idx_power)) {
            result.rejected_records++;
            continue;
        }

        // Parse date and time
        int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0;
        char date_buf[12] = {0};
        char time_buf[10] = {0};

        if (idx_timestamp != -1 && static_cast<size_t>(idx_timestamp) < tokens.size()) {
            // Format: "YYYY-MM-DD HH:MM:SS" or "YYYY/MM/DD HH:MM:SS"
            const std::string& ts = tokens[idx_timestamp];
            if (sscanf(ts.c_str(), "%d-%d-%d %d:%d:%d", &year, &month, &day, &hour, &minute, &second) != 6 &&
                sscanf(ts.c_str(), "%d/%d/%d %d:%d:%d", &year, &month, &day, &hour, &minute, &second) != 6 &&
                sscanf(ts.c_str(), "%d-%d-%d %d:%d", &year, &month, &day, &hour, &minute) != 5) {
                // Try DD/MM/YYYY format
                if (sscanf(ts.c_str(), "%d/%d/%d %d:%d:%d", &day, &month, &year, &hour, &minute, &second) != 6) {
                    result.rejected_records++;
                    continue;
                }
            }
        } else if (idx_date != -1 && idx_time != -1 &&
                   static_cast<size_t>(idx_date) < tokens.size() &&
                   static_cast<size_t>(idx_time) < tokens.size()) {
            const std::string& d_str = tokens[idx_date];
            const std::string& t_str = tokens[idx_time];
            if (d_str.find('/') != std::string::npos) {
                if (sscanf(d_str.c_str(), "%d/%d/%d", &day, &month, &year) != 3) {
                    result.rejected_records++;
                    continue;
                }
            } else if (d_str.find('-') != std::string::npos) {
                if (sscanf(d_str.c_str(), "%d-%d-%d", &year, &month, &day) != 3) {
                    result.rejected_records++;
                    continue;
                }
            }
            if (sscanf(t_str.c_str(), "%d:%d:%d", &hour, &minute, &second) < 2) {
                result.rejected_records++;
                continue;
            }
        } else {
            result.rejected_records++;
            continue;
        }

        // Validate timestamp ranges
        if (year < 1900 || month < 1 || month > 12 || day < 1 || day > 31 ||
            hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 59) {
            result.rejected_records++;
            continue;
        }

        snprintf(date_buf, sizeof(date_buf), "%04d-%02d-%02d", year, month, day);
        snprintf(time_buf, sizeof(time_buf), "%02d:%02d:%02d", hour, minute, second);

        // Power Consumption (kW)
        const std::string& p_str = tokens[idx_power];
        if (p_str.empty() || p_str == "?" || p_str.find('?') != std::string::npos) {
            result.rejected_records++;
            continue;
        }
        char* end_ptr = nullptr;
        double power = std::strtod(p_str.c_str(), &end_ptr);
        if (end_ptr == p_str.c_str() || power < 0.0) {
            result.rejected_records++;
            continue;
        }

        PowerReading rec;
        std::memset(&rec, 0, sizeof(rec));
        rec.timestamp_sec = to_epoch_seconds(year, month, day, hour, minute, second);
        rec.year = year;
        rec.month = month;
        rec.day = day;
        rec.hour = hour;
        rec.minute = minute;
        rec.second = second;
        rec.power_consumption = power;

        strncpy(rec.date_str, date_buf, sizeof(rec.date_str) - 1);
        strncpy(rec.time_str, time_buf, sizeof(rec.time_str) - 1);

        // Optional telemetry fields with safe defaults
        auto parse_double_col = [&](int col_idx, double default_val) -> double {
            if (col_idx >= 0 && static_cast<size_t>(col_idx) < tokens.size()) {
                const auto& s = tokens[col_idx];
                if (!s.empty() && s != "?") {
                    char* ep = nullptr;
                    double v = std::strtod(s.c_str(), &ep);
                    if (ep != s.c_str()) return v;
                }
            }
            return default_val;
        };

        auto parse_int_col = [&](int col_idx, int default_val) -> int {
            if (col_idx >= 0 && static_cast<size_t>(col_idx) < tokens.size()) {
                const auto& s = tokens[col_idx];
                if (!s.empty() && s != "?") {
                    return std::atoi(s.c_str());
                }
            }
            return default_val;
        };

        rec.voltage = parse_double_col(idx_voltage, 230.0);
        rec.current = parse_double_col(idx_current, (rec.voltage > 0) ? (power * 1000.0 / rec.voltage) : 0.0);
        rec.reactive_power = parse_double_col(idx_reactive, 0.0);
        rec.power_factor = parse_double_col(idx_power_factor, 0.95);
        rec.solar_power = parse_double_col(idx_solar, 0.0);
        rec.wind_power = parse_double_col(idx_wind, 0.0);
        rec.grid_supply = parse_double_col(idx_grid_supply, 0.0);
        rec.voltage_fluctuation = parse_double_col(idx_volt_fluct, 0.0);
        rec.overload_condition = parse_int_col(idx_overload, 0);
        rec.transformer_fault = parse_int_col(idx_fault, 0);
        rec.temperature = parse_double_col(idx_temp, 20.0);
        rec.humidity = parse_double_col(idx_humidity, 50.0);
        rec.electricity_price = parse_double_col(idx_price, 0.15);
        rec.predicted_load = parse_double_col(idx_pred_load, power);

        result.records.push_back(rec);
        result.valid_records++;
    }

    // Detect sampling interval from first two records
    if (result.records.size() >= 2) {
        int64_t diff_sec = result.records[1].timestamp_sec - result.records[0].timestamp_sec;
        if (diff_sec > 0 && diff_sec <= 3600) {
            result.detected_interval_minutes = static_cast<double>(diff_sec) / 60.0;
        }
    }

    result.success = true;
    return result;
}

bool DatasetLoader::generate_sample_dataset(const std::string& output_path, int num_days) {
    std::ofstream out(output_path);
    if (!out.is_open()) return false;

    // Header matching Kaggle smart-grid-real-time-load-monitoring-dataset
    out << "Timestamp,Voltage (V),Current (A),Power Consumption (kW),Reactive Power (kVAR),Power Factor,"
        << "Solar Power (kW),Wind Power (kW),Grid Supply (kW),Voltage Fluctuation (%),Overload Condition,"
        << "Transformer Fault,Temperature (°C),Humidity (%),Electricity Price (USD/kWh),Predicted Load (kW)\n";

    std::mt19937_64 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    int start_year = 2024;
    int start_month = 1;
    int start_day = 1;

    // 15-minute intervals -> 4 readings per hour -> 96 readings per day
    int readings_per_day = 96;
    int total_readings = num_days * readings_per_day;

    int cur_year = start_year;
    int cur_month = start_month;
    int cur_day = start_day;

    for (int i = 0; i < total_readings; ++i) {
        int minutes_into_day = (i % readings_per_day) * 15;
        int hour = minutes_into_day / 60;
        int minute = minutes_into_day % 60;

        if (minutes_into_day == 0 && i > 0) {
            cur_day++;
            if (cur_day > 31) {
                cur_day = 1;
                cur_month++;
            }
        }

        // Simulate diurnal smart grid load
        double base_load = 5.0;
        if (hour >= 6 && hour <= 9) base_load = 8.5;       // Morning peak
        else if (hour >= 10 && hour <= 16) base_load = 6.2; // Midday solar offset
        else if (hour >= 18 && hour <= 21) base_load = 10.2;// Evening peak demand
        else if (hour >= 23 || hour <= 4) base_load = 2.5;  // Night baseload

        double power = base_load + noise(rng);
        if (power < 1.0) power = 1.0;

        // Injected extreme peak spikes
        int overload = 0;
        int fault = 0;
        if (i == 150) {
            power = 12.05; // Peak spike 1
            overload = 1;
        } else if (i == 400) {
            power = 12.10; // Grand peak spike
            overload = 1;
            fault = 1;
        } else if (power > 10.5) {
            overload = 1;
        }

        double voltage = 230.0 + noise(rng) * 2.0;
        double current = (power * 1000.0) / voltage;
        double reactive = power * 0.25;
        double pf = 0.92;
        double solar = (hour >= 7 && hour <= 17) ? (15.0 * std::sin((hour - 7) * 3.14159 / 10.0)) : 0.0;
        double wind = 10.0 + noise(rng) * 2.0;
        double volt_fluct = ((voltage - 230.0) / 230.0) * 100.0;
        double temp = 22.0 + 8.0 * std::sin((hour - 6) * 3.14159 / 12.0);
        double humidity = 55.0 - 15.0 * std::sin((hour - 6) * 3.14159 / 12.0);
        double price = 0.20 + (power / 20.0);
        double pred_load = power + (noise(rng) * 0.1);

        out << std::setfill('0') << std::setw(4) << cur_year << "-"
            << std::setfill('0') << std::setw(2) << cur_month << "-"
            << std::setfill('0') << std::setw(2) << cur_day << " "
            << std::setfill('0') << std::setw(2) << hour << ":"
            << std::setfill('0') << std::setw(2) << minute << ":00,"
            << std::fixed << std::setprecision(4)
            << voltage << ","
            << current << ","
            << power << ","
            << reactive << ","
            << pf << ","
            << solar << ","
            << wind << ","
            << 0.0 << ","
            << volt_fluct << ","
            << overload << ","
            << fault << ","
            << temp << ","
            << humidity << ","
            << price << ","
            << pred_load << "\n";
    }

    return true;
}
