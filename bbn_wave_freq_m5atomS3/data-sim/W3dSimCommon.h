#pragma once

#ifndef EIGEN_NON_ARDUINO
#define EIGEN_NON_ARDUINO
#endif

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "WaveFilesSupport.h"
#include "FrameConversions.h"
#include "WaveDirectionDetector.h"

using Eigen::Vector2f;
using Eigen::Vector3f;

inline float wrapDeg(float a) {
    a = std::fmod(a + 180.0f, 360.0f);
    if (a < 0) a += 360.0f;
    return a - 180.0f;
}

inline float diffDeg(float est_deg, float ref_deg) {
    return wrapDeg(est_deg - ref_deg);
}


struct ImuNoiseModel {
    std::mt19937 rng;
    std::normal_distribution<float> w;
    std::normal_distribution<float> n01;
    Vector3f bias0;
    Vector3f bias_rw;
    float sigma_bias_rw = 0.0f;
};

ImuNoiseModel make_imu_noise_model(float sigma_white,
                                   float bias_half_range,
                                   float sigma_bias_rw,
                                   unsigned seed);

Vector3f apply_imu_noise(const Vector3f& truth, ImuNoiseModel& m, float dt);

struct MagNoiseModel {
    std::mt19937 rng;
    std::normal_distribution<float> w_uT;
    std::normal_distribution<float> n01;

    Vector3f bias0_uT;
    Vector3f bias_rw_uT;
    float sigma_bias_rw_uT_sqrt_s = 0.0f;

    Eigen::Matrix3f Mis;
};

MagNoiseModel make_mag_noise_model(float sigma_white_uT,
                                   float bias_residual_range_uT,
                                   float sigma_bias_rw_uT_sqrt_s,
                                   float scale_err_max,
                                   float cross_axis_max,
                                   float misalign_deg_max,
                                   unsigned seed);

Vector3f apply_mag_noise(const Vector3f& ideal_mag_uT_body, MagNoiseModel& m, float dt_mag);

template <class MekfT>
inline Vector3f get_mag_bias_est_uT(const MekfT& mekf)
{
    if constexpr (requires { mekf.get_mag_bias_uT(); }) {
        return mekf.get_mag_bias_uT();
    } else if constexpr (requires { mekf.get_mag_bias(); }) {
        return mekf.get_mag_bias();
    } else if constexpr (requires { mekf.magnetometer_bias_uT(); }) {
        return mekf.magnetometer_bias_uT();
    } else if constexpr (requires { mekf.magnetometer_bias(); }) {
        return mekf.magnetometer_bias();
    } else {
        return Vector3f::Zero();
    }
}

struct DirectionTelemetry {
    float phase = NAN;
    float direction_deg = NAN;
    float direction_deg_generator_signed = NAN;
    float uncertainty_deg = NAN;
    float confidence = NAN;
    float amplitude = NAN;
    Vector2f direction_vec = Vector2f::Zero();
    Vector2f filtered_signal = Vector2f::Zero();
    WaveDirection sign = UNCERTAIN;
    int sign_num = 0;
};

struct FilterSnapshot {
    Vector3f disp_est_zu = Vector3f::Zero();
    Vector3f vel_est_zu = Vector3f::Zero();
    Vector3f acc_est_zu = Vector3f::Zero();
    Vector3f euler_nautical_deg = Vector3f::Zero();

    Vector3f acc_bias_est_ned = Vector3f::Zero();
    Vector3f gyro_bias_est_ned = Vector3f::Zero();
    Vector3f mag_bias_est_ned_uT = Vector3f::Zero();

    float tau_target = NAN;
    float sigma_target = NAN;
    float tuning_target = NAN;
    float tau_applied = NAN;
    float sigma_applied = NAN;
    float tuning_applied = NAN;
    float freq_hz = NAN;
    float period_sec = NAN;
    float accel_variance = NAN;
    float displacement_scale_m = NAN;
    float velocity_scale_mps = NAN;

    DirectionTelemetry direction;
};

class IW3dFusionAdapter {
public:
    virtual ~IW3dFusionAdapter() = default;
    virtual void updateMag(const Vector3f& mag_body_ned) = 0;
    virtual void update(float dt,
                        const Vector3f& gyr_meas_ned,
                        const Vector3f& acc_meas_ned,
                        float temperature_c) = 0;
    virtual FilterSnapshot snapshot() const = 0;
};

using ImuNoiseInjector = std::function<void(Vector3f& acc_body_zu,
                                            Vector3f& gyr_body_zu,
                                            float dt)>;
using MagNoiseInjector = std::function<void(Vector3f& mag_body_enu, float dt_mag)>;

struct SimulationNoiseModels {
    std::optional<ImuNoiseModel> accel_noise;
    std::optional<ImuNoiseModel> gyro_noise;
    std::optional<MagNoiseModel> mag_noise;
    std::vector<ImuNoiseInjector> extra_imu_noise_models;
    std::vector<MagNoiseInjector> extra_mag_noise_models;
};

struct W3dSimulationOptions {
    float dt = 0.005f;
    bool with_mag = true;
    bool add_noise = true;
    float mag_odr_hz = 100.0f;
    float temperature_c = 35.0f;
    std::string output_suffix_with_mag = "_fusion";
    std::string output_suffix_no_mag = "_fusion_nomag";
};

struct W3dSimulationRunResult {
    std::string input_name;
    std::string output_name;
    WaveType wave_type = WaveType::JONSWAP;
    WaveParameters wave_params{};

    std::vector<float> errs_x, errs_y, errs_z, errs_roll, errs_pitch, errs_yaw;
    std::vector<float> ref_x, ref_y, ref_z;
    std::vector<float> accb_err_x, accb_err_y, accb_err_z;
    std::vector<float> gyrb_err_x, gyrb_err_y, gyrb_err_z;
    std::vector<float> magb_err_x, magb_err_y, magb_err_z;
    std::vector<float> accb_true_x, accb_true_y, accb_true_z;
    std::vector<float> gyrb_true_x, gyrb_true_y, gyrb_true_z;
    std::vector<float> magb_true_x, magb_true_y, magb_true_z;
    std::vector<float> freq_hist;
    std::vector<float> dir_deg_hist, dir_unc_hist, dir_conf_hist, dir_amp_hist, dir_phase_hist;
    std::vector<int> dir_sign_num_hist;

    float final_tau_target = NAN;
    float final_sigma_target = NAN;
    float final_tuning_target = NAN;
    float final_tau_applied = NAN;
    float final_sigma_applied = NAN;
    float final_tuning_applied = NAN;
    float final_freq_hz = NAN;
    float final_period_sec = NAN;
    float final_accel_variance = NAN;
};

class W3dSimulationRunner {
public:
    W3dSimulationRunner(W3dSimulationOptions options,
                        SimulationNoiseModels noise_models,
                        IW3dFusionAdapter& fusion_adapter);

    std::optional<W3dSimulationRunResult> run(const std::string& filename);

private:
    std::string make_output_name(const std::string& filename) const;

    W3dSimulationOptions options_;
    SimulationNoiseModels noise_models_;
    IW3dFusionAdapter& fusion_adapter_;
};
