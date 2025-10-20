/*

  Generate simulated waves data

  Copyright 2025, Mikhail Grushinskiy

*/

#include <iostream>

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "TrochoidalWave.h"
#include "FentonWaveVectorized.h"
#include "CnoidalWave.h"
#include "DirectionalSpread.h"
#include "Jonswap3dStokesWaves.h"
#include "PiersonMoskowitzStokes3D_Waves.h"
#include "WaveFilesSupport.h"

// Experiment Config
static constexpr float SAMPLE_RATE_HZ  = 240.0f;
static constexpr float DELTA_T         = 1.0f / SAMPLE_RATE_HZ;
static constexpr float TEST_DURATION_S = 20 * 60.0f;    // 20 minutes

// Example test cases
const std::vector<WaveParameters> waveParamsList = {
    {3.0f,   0.27f, static_cast<float>(M_PI/3.0), 25.0f},
    {5.7f,   1.5f,  static_cast<float>(M_PI/1.5), 25.0f},
    {8.5f,   4.0f,  static_cast<float>(M_PI/6.0), 25.0f},
    {11.4f,  8.5f,  static_cast<float>(M_PI/2.5), 25.0f}
};

// Shared Fill Helpers
template<typename State>
static void fill_wave_sample_from_state(Wave_Sample &dst, const State &st) {
    dst.disp_x = static_cast<float>(st.displacement.x());
    dst.disp_y = static_cast<float>(st.displacement.y());
    dst.disp_z = static_cast<float>(st.displacement.z());
    dst.vel_x  = static_cast<float>(st.velocity.x());
    dst.vel_y  = static_cast<float>(st.velocity.y());
    dst.vel_z  = static_cast<float>(st.velocity.z());
    dst.acc_x  = static_cast<float>(st.acceleration.x());
    dst.acc_y  = static_cast<float>(st.acceleration.y());
    dst.acc_z  = static_cast<float>(st.acceleration.z());
}

template<typename IMU>
static void fill_imu_sample_from_readings(IMU_Sample &dst, const IMU &imu) {
    dst.acc_bx = static_cast<float>(imu.accel_body.x());
    dst.acc_by = static_cast<float>(imu.accel_body.y());
    dst.acc_bz = static_cast<float>(imu.accel_body.z());
    dst.gyro_x = static_cast<float>(imu.gyro_body.x());
    dst.gyro_y = static_cast<float>(imu.gyro_body.y());
    dst.gyro_z = static_cast<float>(imu.gyro_body.z());
}

// Default IMU filler (zeros)
static void fill_default_imu(IMU_Sample &imu) {
    imu = {}; // zero all fields
}

template<typename Model>
static void export_spectrum(const WaveParameters &wp, WaveType type, Model &model, int N_theta = 72)
{
    auto freqs = model.frequencies();
    Eigen::MatrixXd E = model.getDirectionalSpectrum(N_theta);

    std::string spec_filename = WaveFileNaming::generate(FileKind::Spectrum, type, wp);
    WaveSpectrumCSVWriter writer(spec_filename);

    const double dtheta = 360.0 / N_theta;
    for (int i = 0; i < freqs.size(); ++i) {
        for (int m = 0; m < N_theta; ++m) {
            double theta_deg = -180.0 + m * dtheta;
            writer.write(freqs(i), theta_deg, E(i, m));
        }
    }
    writer.close();
    std::cout << "Wrote " << spec_filename << "\n";
}

// Sampling Helpers
static Wave_Data_Sample sample_gerstner(double t, TrochoidalWave<float> &wave_obj) {
    Wave_Data_Sample out{};
    out.time = t;

    // Reference initial particle location (x0, z0 at the surface)
    float x0 = 0.0f;
    float z0 = 0.0f;

    // Lagrangian absolute positions
    float x = wave_obj.horizontalPosition(x0, z0, static_cast<float>(t));
    float z = wave_obj.verticalPosition  (x0, z0, static_cast<float>(t));

    // Lagrangian velocities
    float u = wave_obj.horizontalVelocity(x0, z0, static_cast<float>(t));
    float w = wave_obj.verticalVelocity  (x0, z0, static_cast<float>(t));

    // Lagrangian accelerations
    float ax = wave_obj.horizontalAcceleration(x0, z0, static_cast<float>(t));
    float az = wave_obj.verticalAcceleration  (x0, z0, static_cast<float>(t));

    // Write **displacements** (relative to (x0,z0)), not absolute positions
    out.wave.disp_x = x - x0;
    out.wave.disp_y = 0.0f;   // 2D Gerstner
    out.wave.disp_z = z - z0;

    out.wave.vel_x  = u;
    out.wave.vel_y  = 0.0f;
    out.wave.vel_z  = w;

    out.wave.acc_x  = ax;
    out.wave.acc_y  = 0.0f;
    out.wave.acc_z  = az;

    fill_default_imu(out.imu);
    return out;
}

template<int N=128>
static Wave_Data_Sample sample_jonswap(double t, Jonswap3dStokesWaves<N> &model) {
    Wave_Data_Sample out{};
    out.time = t;

    // Lagrangian state at sensor depth z=0
    auto state = model.getLagrangianState(0.0, 0.0, t, 0.0);
    fill_wave_sample_from_state(out.wave, state);

    // IMU (buoy-attached) – use correct dt for angular velocity
    auto imu = model.getIMUReadings(0.0, 0.0, t, 0.0, DELTA_T);
    fill_imu_sample_from_readings(out.imu, imu);

    // Reference Euler at advected buoy (rotationMatrixAt already advects)
    Eigen::Vector3d euler = model.getEulerAngles(0.0, 0.0, t);

    out.imu.roll_deg  = static_cast<float>(euler.x());
    out.imu.pitch_deg = static_cast<float>(euler.y());
    out.imu.yaw_deg   = static_cast<float>(euler.z());

    return out;
}

template<int N=128, int ORDER=3>
static Wave_Data_Sample sample_pmstokes(double t, PMStokesN3dWaves<N, ORDER> &model) {
    Wave_Data_Sample out{};
    out.time = t;

    // Lagrangian surface particle (buoy)
    auto state = model.getLagrangianState(t);
    fill_wave_sample_from_state(out.wave, state);

    // IMU (buoy-attached) – use correct dt
    auto imu = model.getIMUReadings(0.0, 0.0, t, 0.0, DELTA_T);
    fill_imu_sample_from_readings(out.imu, imu);

    // Reference Euler (no manual displacement)
    Eigen::Vector3d euler = model.getEulerAngles(0.0, 0.0, t);

    out.imu.roll_deg  = static_cast<float>(euler.x());
    out.imu.pitch_deg = static_cast<float>(euler.y());
    out.imu.yaw_deg   = static_cast<float>(euler.z());

    return out;
}

template<int ORDER=5>
static std::vector<Wave_Data_Sample> sample_fenton(
        const WaveParameters &wp, double duration, double dt) 
{
    std::vector<Wave_Data_Sample> results;

    auto fenton_params = FentonWave<ORDER>::infer_fenton_parameters_from_amplitude(
        wp.height / 2.0f, 200.0f, 2.0f * M_PI / wp.period, wp.phase);

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
        fill_default_imu(out.imu);
        results.push_back(out);
    };
    fenton_tracker.track_floating_object(duration, dt, callback);
    return results;
}

static Wave_Data_Sample sample_cnoidal(double t, CnoidalWave<float> &wave) {
    Wave_Data_Sample out{};
    out.time = t;
    auto state = wave.getLagrangianState(0.0f, 0.0f, t);
    fill_wave_sample_from_state(out.wave, state);
    fill_default_imu(out.imu); 
    return out;
}

// Scenario Runner
static void run_one_scenario(WaveType waveType, const WaveParameters &wp) {
    WaveParameters wp_copy = wp;
    if (waveType == WaveType::GERSTNER || 
        waveType == WaveType::FENTON   || 
        waveType == WaveType::CNOIDAL) {
        wp_copy.direction = 0.0f;
    }
    std::string filename = WaveFileNaming::generate(FileKind::Data, waveType, wp_copy);

    WaveDataCSVWriter writer(filename);
    writer.write_header();

    double sim_t = 0.0;
    int total_steps = static_cast<int>(std::round(TEST_DURATION_S / DELTA_T));

    if (waveType == WaveType::GERSTNER) {
        TrochoidalWave<float> trocho(wp.height / 2.0f, wp.period, wp.phase);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_gerstner(sim_t, trocho);
            writer.write(samp);
            sim_t += DELTA_T;
        }
        writer.close();
    }
    else if (waveType == WaveType::JONSWAP) {
        auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
            wp.direction * M_PI / 180.0, 10.0, GLOBAL_SEED);
        auto jonswap_model = std::make_unique<Jonswap3dStokesWaves<128>>(
            wp.height, wp.period, dirDist, 0.02, 0.8, 3.3, g_std, GLOBAL_SEED);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_jonswap(sim_t, *jonswap_model);
            writer.write(samp);
            sim_t += DELTA_T;
        }
        writer.close();
        export_spectrum(wp, WaveType::JONSWAP, *jonswap_model);
    }
    else if (waveType == WaveType::FENTON) {
        auto samples = sample_fenton<5>(wp, TEST_DURATION_S, DELTA_T);
        for (auto &samp : samples) writer.write(samp);
        writer.close();
    }
    else if (waveType == WaveType::PMSTOKES) {
        auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
            wp.direction * M_PI / 180.0, 10.0, GLOBAL_SEED);
        PMStokesN3dWaves<128, 3> waveModel(
            wp.height, wp.period, dirDist, 0.02, 0.8, g_std, GLOBAL_SEED);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_pmstokes(sim_t, waveModel);
            writer.write(samp);
            sim_t += DELTA_T;
        }
        writer.close();
        export_spectrum(wp, WaveType::PMSTOKES, waveModel);
    }
    else if (waveType == WaveType::CNOIDAL) {
        CnoidalWave<float> cnoidal(200.0f, wp.height / 2.0f, wp.period, 0.0f, g_std);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_cnoidal(sim_t, cnoidal);
            writer.write(samp);
            sim_t += DELTA_T;
        }
        writer.close();
    }
    std::cout << "Wrote " << filename << "\n";
}

static void run_all_wave_types(const WaveParameters &wp, int idx = -1) {
    for (WaveType wt : {WaveType::GERSTNER, WaveType::JONSWAP,
                        WaveType::FENTON, WaveType::PMSTOKES,
                        WaveType::CNOIDAL}) {
        run_one_scenario(wt, wp);
    }
    if (idx >= 0) {
        std::cout << "Wave index " << idx << " complete.\n";
    }
}

// Main
int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [wave_index]\n";
        return 1;
    }
    if (argc == 2) {
        int idx = std::stoi(argv[1]);
        if (idx < 0 || idx >= static_cast<int>(waveParamsList.size())) {
            std::cerr << "Invalid wave_index " << idx
                      << " (must be 0.." << (waveParamsList.size() - 1) << ")\n";
            return 1;
        }
        run_all_wave_types(waveParamsList[idx], idx);
    } else {
        for (size_t idx = 0; idx < waveParamsList.size(); ++idx) {
            run_all_wave_types(waveParamsList[idx], static_cast<int>(idx));
        }
        std::cout << "All wave data generation complete.\n";
    }
    return 0;
}
