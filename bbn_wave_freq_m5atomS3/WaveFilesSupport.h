#pragma once
#include <cmath>
#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>
#include <optional>
#include <vector>
#include <fstream>
#include <stdexcept>

/*
  Copyright 2025, Mikhail Grushinskiy
*/

static constexpr unsigned GLOBAL_SEED  = 42u;           // global seed for reproducibility

// Wave Types
enum class WaveType { GERSTNER=0, JONSWAP=1, FENTON=2, PMSTOKES=3, CNOIDAL=4 };

// Wave Parameters
struct WaveParameters {
    float period;     // wave period in seconds
    float height;     // wave height in m
    float phase;      // initial phase in radians
    float direction;  // azimuth in degrees
};

// Data Structures for Samples
struct Wave_Sample {
    float disp_x{}, disp_y{}, disp_z{};
    float vel_x{}, vel_y{}, vel_z{};
    float acc_x{}, acc_y{}, acc_z{};
};

struct IMU_Sample {
    float acc_bx{}, acc_by{}, acc_bz{};
    float gyro_x{}, gyro_y{}, gyro_z{};
    float roll_deg{}, pitch_deg{}, yaw_deg{};
};

struct Wave_Data_Sample {
    double time{};   // simulation time
    Wave_Sample wave{};
    IMU_Sample imu{};
};

// Data structure for one spectrum bin
struct WaveSpectrumRecord {
    double f_Hz{};       // frequency in Hz
    double theta_deg{};  // direction in degrees
    double E{};          // spectral density
};

// Wave File Naming
class WaveFileNaming {
public:
    struct ParsedName {
        WaveType type{};
        double height{};
        double length{};
        double azimuth{};
        double phaseDeg{};
    };

    static std::string generate(WaveType type, const WaveParameters &wp) {
        double length = (wp.period > 0.0)
                      ? (g_std * wp.period * wp.period / (2 * M_PI))  // deep-water approx
                      : 0.0;
        double phaseDeg = wp.phase * 180.0 / M_PI;

        std::ostringstream oss;
        oss << "wave_data_" << to_string(type)
            << "_H" << std::fixed << std::setprecision(3) << wp.height
            << "_L" << std::fixed << std::setprecision(3) << length
            << "_A" << std::fixed << std::setprecision(2) << wp.direction
            << "_P" << std::fixed << std::setprecision(2) << phaseDeg
            << ".csv";
        return oss.str();
    }

    static std::optional<ParsedName> parse(const std::string &filename) {
        ParsedName result{};
        std::string stem = filename;
        auto posSlash = stem.find_last_of("/\\");
        if (posSlash != std::string::npos) stem = stem.substr(posSlash + 1);

        if (!stem.starts_with("wave_data_") || !stem.ends_with(".csv")) {
            return std::nullopt;
        }
        stem = stem.substr(10, stem.size() - 14); // strip prefix/suffix
        auto tokens = split(stem, '_');
        if (tokens.size() < 5) return std::nullopt;

        auto optType = from_string(tokens[0]);
        if (!optType) return std::nullopt;
        result.type = *optType;

        try {
            for (size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i].starts_with("H")) result.height = std::stod(tokens[i].substr(1));
                else if (tokens[i].starts_with("L")) result.length = std::stod(tokens[i].substr(1));
                else if (tokens[i].starts_with("A")) result.azimuth = std::stod(tokens[i].substr(1));
                else if (tokens[i].starts_with("P")) result.phaseDeg = std::stod(tokens[i].substr(1));
            }
        } catch (...) {
            return std::nullopt;
        }
        return result;
    }

    static std::optional<std::pair<WaveType, WaveParameters>> parse_to_params(const std::string &filename) {
        auto parsed = parse(filename);
        if (!parsed) return std::nullopt;

        WaveParameters wp{};
        wp.height    = static_cast<float>(parsed->height);
        wp.direction = static_cast<float>(parsed->azimuth);
        wp.phase     = static_cast<float>(parsed->phaseDeg * M_PI / 180.0);

        if (parsed->length > 0.0) {
            double T = std::sqrt(parsed->length / g_std * 2 * M_PI);
            wp.period = static_cast<float>(T);
        } else {
            wp.period = 0.0f;
        }
        return std::make_pair(parsed->type, wp);
    }

private:
    static std::string to_string(WaveType type) {
        switch (type) {
            case WaveType::GERSTNER:  return "gerstner";
            case WaveType::JONSWAP:   return "jonswap";
            case WaveType::FENTON:    return "fenton";
            case WaveType::PMSTOKES:  return "pmstokes";
            case WaveType::CNOIDAL:   return "cnoidal";
        }
        return "unknown";
    }

    static std::optional<WaveType> from_string(const std::string &name) {
        if (name == "gerstner") return WaveType::GERSTNER;
        if (name == "jonswap")  return WaveType::JONSWAP;
        if (name == "fenton")   return WaveType::FENTON;
        if (name == "pmstokes") return WaveType::PMSTOKES;
        if (name == "cnoidal")  return WaveType::CNOIDAL;
        return std::nullopt;
    }

    static std::vector<std::string> split(const std::string &s, char delim) {
        std::vector<std::string> elems;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) elems.push_back(item);
        return elems;
    }
};

// CSV Reader
class WaveDataCSVReader {
public:
    explicit WaveDataCSVReader(const std::string &filename) : ifs(filename) {
        if (!ifs.is_open()) throw std::runtime_error("Failed to open " + filename);
        std::string header; std::getline(ifs, header); // skip header
    }

    template<typename Callback>
    void for_each_record(Callback cb) {
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            Wave_Data_Sample rec{};
            if (read_csv_record(line, rec)) cb(rec);
        }
    }

    void close() { if (ifs.is_open()) ifs.close(); }

private:
    std::ifstream ifs;

    static bool read_csv_record(const std::string &line, Wave_Data_Sample &s) {
        std::istringstream iss(line);
        char comma;

        iss >> s.time >> comma
            >> s.wave.disp_x >> comma >> s.wave.disp_y >> comma >> s.wave.disp_z >> comma
            >> s.wave.vel_x  >> comma >> s.wave.vel_y  >> comma >> s.wave.vel_z  >> comma
            >> s.wave.acc_x  >> comma >> s.wave.acc_y  >> comma >> s.wave.acc_z  >> comma
            >> s.imu.acc_bx  >> comma >> s.imu.acc_by  >> comma >> s.imu.acc_bz  >> comma
            >> s.imu.gyro_x  >> comma >> s.imu.gyro_y  >> comma >> s.imu.gyro_z  >> comma
            >> s.imu.roll_deg >> comma >> s.imu.pitch_deg >> comma >> s.imu.yaw_deg;

        return static_cast<bool>(iss);
    }     
};

// CSV Writer
class WaveDataCSVWriter {
public:
    explicit WaveDataCSVWriter(const std::string &filename, bool append = false) {
        if (append) ofs.open(filename, std::ios::app);
        else ofs.open(filename, std::ios::trunc);
        if (!ofs.is_open()) throw std::runtime_error("Failed to open " + filename);
    }

    void write_header() {
        ofs << "time,"
            << "disp_x,disp_y,disp_z,"
            << "vel_x,vel_y,vel_z,"
            << "acc_x,acc_y,acc_z,"
            << "acc_bx,acc_by,acc_bz,"
            << "gyro_x,gyro_y,gyro_z,"
            << "roll_deg,pitch_deg,yaw_deg\n";
    }

    void write(const Wave_Data_Sample &s) {
        ofs << s.time << ","
            << s.wave.disp_x << "," << s.wave.disp_y << "," << s.wave.disp_z << ","
            << s.wave.vel_x  << "," << s.wave.vel_y  << "," << s.wave.vel_z << ","
            << s.wave.acc_x  << "," << s.wave.acc_y  << "," << s.wave.acc_z << ","
            << s.imu.acc_bx  << "," << s.imu.acc_by  << "," << s.imu.acc_bz << ","
            << s.imu.gyro_x  << "," << s.imu.gyro_y  << "," << s.imu.gyro_z << ","
            << s.imu.roll_deg << "," << s.imu.pitch_deg << "," << s.imu.yaw_deg
            << "\n";
    }

    void flush() { ofs.flush(); }
    void close() { if (ofs.is_open()) ofs.close(); }

private:
    std::ofstream ofs;
};

class WaveSpectrumCSVWriter {
public:
    explicit WaveSpectrumCSVWriter(const std::string &filename) : ofs(filename) {
        if (!ofs.is_open()) throw std::runtime_error("Failed to open " + filename);
        ofs << "f_Hz,theta_deg,E\n";
    }
    void write(double f, double theta_deg, double E) {
        ofs << f << "," << theta_deg << "," << E << "\n";
    }
    void close() { if (ofs.is_open()) ofs.close(); }
private:
    std::ofstream ofs;
};
