#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <limits>
#include <iomanip>
#include <memory>
#include <stdexcept>

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "TrochoidalWave.h"
#include "Jonswap3dStokesWaves.h"
#include "FentonWaveVectorized.h"
#include "PiersonMoskowitzStokes3D_Waves.h"
#include "DirectionalSpread.h"

// Config
static constexpr float SAMPLE_RATE_HZ  = 240.0f;
static constexpr float DELTA_T         = 1.0f / SAMPLE_RATE_HZ;
static constexpr float TEST_DURATION_S = 10 * 60.0f;
static constexpr float g_std           = 9.80665f;

enum class WaveType { GERSTNER=0, JONSWAP=1, FENTON=2, PMSTOKES=3 };

struct WaveParameters {
    float freqHz;
    float height;
    float phase;
    float direction;
};

const std::vector<WaveParameters> waveParamsList = {
    {1.0f/3.0f,  0.135f, static_cast<float>(M_PI/3.0), 30.0f},
    {1.0f/5.7f,  0.75f,  static_cast<float>(M_PI/3.0), 30.0f},
    {1.0f/8.5f,  2.0f,   static_cast<float>(M_PI/3.0), 30.0f},
    {1.0f/11.4f, 4.25f,  static_cast<float>(M_PI/3.0), 30.0f},
    {1.0f/14.3f, 7.4f,   static_cast<float>(M_PI/3.0), 30.0f}
};

// --- Data structures ---
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

// --- CSV Reader ---
static bool read_csv_record(const std::string &line, Wave_Data_Sample &s) {
    std::istringstream iss(line);
    char comma;
    return (
        iss >> s.time >> comma
        >> s.wave.disp_x >> comma >> s.wave.disp_y >> comma >> s.wave.disp_z >> comma
        >> s.wave.vel_x  >> comma >> s.wave.vel_y  >> comma >> s.wave.vel_z >> comma
        >> s.wave.acc_x  >> comma >> s.wave.acc_y  >> comma >> s.wave.acc_z >> comma
        >> s.imu.acc_bx  >> comma >> s.imu.acc_by  >> comma >> s.imu.acc_bz >> comma
        >> s.imu.gyro_x  >> comma >> s.imu.gyro_y  >> comma >> s.imu.gyro_z >> comma
        >> s.imu.roll_deg >> comma >> s.imu.pitch_deg >> comma >> s.imu.yaw_deg
    );
}

class WaveDataCSVReader {
public:
    explicit WaveDataCSVReader(const std::string &filename) : ifs(filename) {
        if (!ifs.is_open()) {
            throw std::runtime_error("Failed to open " + filename);
        }
        std::string header;
        std::getline(ifs, header); // skip header
    }

    template<typename Callback>
    void for_each_record(Callback cb) {
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            Wave_Data_Sample rec{};
            if (read_csv_record(line, rec)) {
                cb(rec);
            }
        }
    }

private:
    std::ifstream ifs;
};

// --- CSV Writer ---
class WaveDataCSVWriter {
public:
    explicit WaveDataCSVWriter(const std::string &filename, bool append = false) {
        if (append) {
            ofs.open(filename, std::ios::app);
        } else {
            ofs.open(filename, std::ios::trunc);
        }
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open " + filename);
        }
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

private:
    std::ofstream ofs;
};

// --- Sampling helpers ---
static Wave_Data_Sample sample_gerstner(double t, TrochoidalWave<float> &wave_obj) {
    Wave_Data_Sample out{};
    out.time        = t;
    out.wave.disp_z = wave_obj.surfaceElevation(static_cast<float>(t));
    out.wave.vel_z  = wave_obj.surfaceVerticalVelocity(static_cast<float>(t));
    out.wave.acc_z  = wave_obj.surfaceVerticalAcceleration(static_cast<float>(t));
    return out;
}

template<int N=128>
static Wave_Data_Sample sample_jonswap(double t, Jonswap3dStokesWaves<N> &model) {
    auto state = model.getLagrangianState(0.0, 0.0, t, 0.0);
    Wave_Data_Sample out{};
    out.time        = t;
    out.wave.disp_z = state.position.z();
    out.wave.vel_z  = state.velocity.z();
    out.wave.acc_z  = state.acceleration.z();
    return out;
}

template<int N=128, int ORDER=3>
static Wave_Data_Sample sample_pmstokes(double t, PMStokesN3dWaves<N, ORDER> &model) {
    auto state = model.getLagrangianState(t);
    Wave_Data_Sample out{};
    out.time        = t;
    out.wave.disp_z = state.position.z();
    out.wave.vel_z  = state.velocity.z();
    out.wave.acc_z  = state.acceleration.z();
    return out;
}

template<int ORDER=4>
static std::vector<Wave_Data_Sample> sample_fenton(
        const WaveParameters &wp,
        double duration,
        double dt) 
{
    std::vector<Wave_Data_Sample> results;

    auto fenton_params = FentonWave<ORDER>::infer_fenton_parameters_from_amplitude(
        wp.height, 200.0f, 2.0f * M_PI * wp.freqHz, wp.phase);

    WaveSurfaceTracker<ORDER> fenton_tracker(
        fenton_params.height,
        fenton_params.depth,
        fenton_params.length,
        fenton_params.initial_x,
        5.0f,    // mass (kg)
        0.1f     // drag coeff
    );

    auto callback = [&](float time, float dt, float elevation,
                        float vertical_velocity, float vertical_acceleration,
                        float x, float vx) {
        (void)dt; (void)x; (void)vx;
        Wave_Data_Sample out{};
        out.time        = time;
        out.wave.disp_z = elevation;
        out.wave.vel_z  = vertical_velocity;
        out.wave.acc_z  = vertical_acceleration;
        results.push_back(out);
    };

    fenton_tracker.track_floating_object(duration, dt, callback);
    return results;
}

// --- Scenario runner ---
static void run_one_scenario(WaveType waveType, const WaveParameters &wp) {
    std::string waveName;
    switch (waveType) {
        case WaveType::GERSTNER:  waveName = "gerstner"; break;
        case WaveType::JONSWAP:   waveName = "jonswap";  break;
        case WaveType::FENTON:    waveName = "fenton";   break;
        case WaveType::PMSTOKES:  waveName = "pmstokes"; break;
    }
    char buf[128];
    std::snprintf(buf, sizeof(buf), "wave_data_%s_h%.3f.csv",
                  waveName.c_str(), wp.height);

    WaveDataCSVWriter writer(buf);
    writer.write_header();

    double sim_t = 0.0;
    int total_steps = static_cast<int>(std::ceil(TEST_DURATION_S * SAMPLE_RATE_HZ));

    if (waveType == WaveType::GERSTNER) {
        float period = 1.0f / wp.freqHz;
        TrochoidalWave<float> trocho(wp.height, period, wp.phase);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_gerstner(sim_t, trocho);
            writer.write(samp);
            sim_t += DELTA_T;
        }
    }
    else if (waveType == WaveType::JONSWAP) {
        float period = 1.0f / wp.freqHz;
        auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
            wp.direction * M_PI / 180.0, 10.0, 42u);
        auto jonswap_model = std::make_unique<Jonswap3dStokesWaves<128>>(
            wp.height, period, dirDist, 0.02, 0.8, 3.3, g_std, 42u);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_jonswap(sim_t, *jonswap_model);
            writer.write(samp);
            sim_t += DELTA_T;
        }
    }
    else if (waveType == WaveType::FENTON) {
        auto samples = sample_fenton<4>(wp, TEST_DURATION_S, DELTA_T);
        for (auto &samp : samples) {
            writer.write(samp);
        }
    }
    else if (waveType == WaveType::PMSTOKES) {
        auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
            wp.direction * M_PI / 180.0, 10.0, 42u);
        PMStokesN3dWaves<128, 3> waveModel(
            wp.height, 1.0f/wp.freqHz, dirDist, 0.02, 0.8, g_std, 42u);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_pmstokes(sim_t, waveModel);
            writer.write(samp);
            sim_t += DELTA_T;
        }
    }

    printf("Wrote %s\n", buf);
}

// --- Main ---
int main() {
    for (const auto& wp : waveParamsList) {
        for (int wt = 0; wt <= static_cast<int>(WaveType::PMSTOKES); ++wt) {
            run_one_scenario(static_cast<WaveType>(wt), wp);
        }
    }
    printf("All wave data generation complete.\n");
    return 0;
}
