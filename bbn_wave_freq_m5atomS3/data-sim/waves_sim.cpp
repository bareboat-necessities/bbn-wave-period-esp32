#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <limits>
#include <iomanip>
#include <memory>

#define EIGEN_NON_ARDUINO

#include "TrochoidalWave.h"
#include "Jonswap3dStokesWaves.h"
#include "FentonWaveVectorized.h"
#include "PiersonMoskowitzStokes3D_Waves.h"

// Config
static constexpr float SAMPLE_RATE_HZ = 240.0f;
static constexpr float DELTA_T = 1.0f / SAMPLE_RATE_HZ;
static constexpr float TEST_DURATION_S = 5 * 60.0f;

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

// --- Sampling helpers ---
struct Wave_Sample {
    // World frame
    float disp_x, disp_y, elevation;
    float vel_x, vel_y, vel_z;
    float acc_x, acc_y, acc_z;
};

static Wave_Sample sample_gerstner(double t, TrochoidalWave<float> &wave_obj) {
    Wave_Sample s;
    s.elevation = wave_obj.surfaceElevation(static_cast<float>(t));
    s.vel_z     = wave_obj.surfaceVerticalVelocity(static_cast<float>(t));
    s.accel_z   = wave_obj.surfaceVerticalAcceleration(static_cast<float>(t));
    return s;
}

template<int N=128>
static Wave_Sample sample_jonswap(double t, Jonswap3dStokesWaves<N> &model) {
    auto state = model.getLagrangianState(0.0, 0.0, t, 0.0);
    Wave_Sample s;
    s.elevation = state.position.z();
    s.vel_z     = state.velocity.z();
    s.accel_z   = state.acceleration.z();
    return s;
}

template<int N=128, int ORDER=3>
static Wave_Sample sample_pmstokes(double t, PMStokesN3dWaves<N, ORDER> &model) {
    auto state = model.getLagrangianState(t);
    Wave_Sample s;
    s.elevation = state.position.z();
    s.vel_z     = state.velocity.z();
    s.accel_z   = state.acceleration.z();
    return s;
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
    std::ofstream ofs(buf);
    if (!ofs.is_open()) {
        fprintf(stderr, "Failed to open %s\n", buf);
        return;
    }

    ofs << "time,elevation,vel_z,accel_z\n";

    double sim_t = 0.0;
    int total_steps = static_cast<int>(std::ceil(TEST_DURATION_S * SAMPLE_RATE_HZ));

    if (waveType == WaveType::GERSTNER) {
        float period = 1.0f / wp.freqHz;
        TrochoidalWave<float> trocho(wp.height, period, wp.phase);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_gerstner(sim_t, trocho);
            ofs << sim_t << "," << samp.elevation << "," << samp.vel_z << "," << samp.accel_z << "\n";
            sim_t += DELTA_T;
        }
    }
    else if (waveType == WaveType::JONSWAP) {
        float period = 1.0f / wp.freqHz;
        auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(wp.direction * M_PI / 180.0, 10.0, 42u);
        auto jonswap_model = std::make_unique<Jonswap3dStokesWaves<128>>(wp.height, period, dirDist, 0.02, 0.8, 3.3, g_std, 42u);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_jonswap(sim_t, *jonswap_model);
            ofs << sim_t << "," << samp.elevation << "," << samp.vel_z << "," << samp.accel_z << "\n";
            sim_t += DELTA_T;
        }
    }
    else if (waveType == WaveType::FENTON) {
        auto fenton_params = FentonWave<4>::infer_fenton_parameters_from_amplitude(wp.height, 200.0f, 2.0f * M_PI * wp.freqHz, wp.phase);
        WaveSurfaceTracker<4> fenton_tracker(
            fenton_params.height,
            fenton_params.depth,
            fenton_params.length,
            fenton_params.initial_x,
            5.0f,    // mass (kg)
            0.1f     // drag coeff
        );
        auto callback = [&](float time, float dt, float elevation, float vertical_velocity, float vertical_acceleration, float x, float vx) {
            ofs << time << "," << elevation << "," << vertical_velocity << "," << vertical_acceleration << "\n";
        };
        fenton_tracker.track_floating_object(TEST_DURATION_S, DELTA_T, callback);
    }
    else if (waveType == WaveType::PMSTOKES) {
        auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(wp.direction * M_PI / 180.0, 10.0, 42u);
        PMStokesN3dWaves<128, 3> waveModel(wp.height, 1.0f/wp.freqHz, dirDist, 0.02, 0.8, g_std, 42u);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_pmstokes(sim_t, waveModel);
            ofs << sim_t << "," << samp.elevation << "," << samp.vel_z << "," << samp.accel_z << "\n";
            sim_t += DELTA_T;
        }
    }

    ofs.close();
    printf("Wrote %s\n", buf);
}

int main() {
    for (const auto& wp : waveParamsList) {
        for (int wt = 0; wt <= static_cast<int>(WaveType::PMSTOKES); ++wt) {
            run_one_scenario(static_cast<WaveType>(wt), wp);
        }
    }
    printf("All wave data generation complete.\n");
    return 0;
}
