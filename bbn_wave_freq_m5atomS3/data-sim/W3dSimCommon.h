#pragma once

#ifndef EIGEN_NON_ARDUINO
#define EIGEN_NON_ARDUINO
#endif

#include <algorithm>
#include <cmath>
#include <numbers>
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

static inline float deg_to_rad(float d) { return d * float(std::numbers::pi_v<float> / 180.0); }
static inline float rad_to_deg(float r) { return r * float(180.0 / std::numbers::pi_v<float>); }

inline float wrapAxialDeg90(float a) {
    a = std::fmod(a + 180.0f, 360.0f);
    if (a < 0) a += 360.0f;
    a -= 180.0f;
    if (a > 90.0f) a -= 180.0f;
    if (a < -90.0f) a += 180.0f;
    return a;
}

inline float dirDegGeneratorSignedFromVec(const Vector2f& v) {
    float deg = rad_to_deg(std::atan2(v.x(), v.y()));
    return wrapAxialDeg90(deg);
}

inline float p0_s_from_sigma_tau(float sigma_a, float tau) {
    if (!std::isfinite(sigma_a) || !std::isfinite(tau)) return NAN;
    return sigma_a * tau * tau;
}

template<typename T>
inline T mean_vec(const std::vector<T>& v) {
    if (v.empty()) return T(NAN);
    T s = 0;
    for (const auto& x : v) s += x;
    return s / T(v.size());
}

template<typename T>
inline T median_vec(std::vector<T> v) {
    if (v.empty()) return T(NAN);
    size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    if (n % 2) return v[n / 2];
    auto lo = *std::max_element(v.begin(), v.begin() + n / 2);
    auto hi = v[n / 2];
    return (lo + hi) / T(2);
}

template<typename T>
inline T percentile_vec(std::vector<T> v, double p01) {
    if (v.empty()) return T(NAN);
    if (p01 <= 0) return *std::min_element(v.begin(), v.end());
    if (p01 >= 1) return *std::max_element(v.begin(), v.end());
    std::sort(v.begin(), v.end());
    double idx = p01 * (v.size() - 1);
    size_t i = size_t(std::floor(idx));
    double frac = idx - double(i);
    if (i + 1 >= v.size()) return v[i];
    return T(v[i] * (1.0 - frac) + v[i + 1] * frac);
}

struct CircStats {
    float mean_deg = NAN;
    float std_deg = NAN;
};

inline CircStats circular_stats_180(const std::vector<float>& degs) {
    CircStats cs;
    if (degs.empty()) return cs;

    double C = 0, S = 0;
    for (float d : degs) {
        const double a2 = 2.0 * deg_to_rad(d);
        C += std::cos(a2);
        S += std::sin(a2);
    }
    C /= double(degs.size());
    S /= double(degs.size());

    const double R = std::sqrt(C * C + S * S);
    const double a2_mean = std::atan2(S, C);

    float md = float(rad_to_deg(0.5 * a2_mean));
    md = wrapAxialDeg90(md);
    cs.mean_deg = md;
    cs.std_deg = (R > 1e-12)
        ? float(rad_to_deg(0.5 * std::sqrt(std::max(0.0, -2.0 * std::log(R)))))
        : 90.0f;
    cs.std_deg = std::min(cs.std_deg, 90.0f);
    return cs;
}

class RMSReport {
public:
    void add(float value) { sum_sq_ += value * value; count_++; }
    float rms() const { return count_ ? std::sqrt(sum_sq_ / float(count_)) : NAN; }

private:
    float sum_sq_ = 0.0f;
    size_t count_ = 0;
};

extern const float g_std;


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
    bool with_mag = true;

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

struct W3dFailureLimits {
    float err_limit_percent_z_jonswap = 0.0f;
    float err_limit_percent_z_pmstokes = 0.0f;
    float err_limit_yaw_deg = 0.0f;
    float err_limit_percent_3d_jonswap = 0.0f;
    float err_limit_percent_3d_pmstokes = 0.0f;
    float acc_z_bias_percent = 0.0f;
    float bias_3d_percent = 0.0f;
};

struct W3dSummaryLabels {
    const char* target = "RS_target";
    const char* applied = "RS_applied";
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

void print_summary_and_fail_if_needed(const W3dSimulationRunResult& result,
                                      float dt,
                                      const W3dFailureLimits& limits,
                                      const W3dSummaryLabels& labels = {});

std::vector<std::string> collect_wave_data_files(const std::filesystem::path& directory);

template <typename AdapterT>
inline std::optional<W3dSimulationRunResult> process_wave_file_for_tracker(const std::string& filename,
                                                                           float dt,
                                                                           bool with_mag,
                                                                           bool add_noise,
                                                                           float mag_odr_hz)
{
    const float acc_sigma = 1.51e-3f * g_std;
    const float gyr_sigma = 0.00157f;
    const float acc_bias_range = 5e-3f * g_std;
    const float gyr_bias_range = 0.05f * float(std::numbers::pi_v<float> / 180.0f);
    const float acc_bias_rw = 0.0005f;
    const float gyr_bias_rw = 0.00001f;
    const float mag_sigma_uT = (mag_odr_hz <= 20.0f) ? 0.30f : 0.60f;

    SimulationNoiseModels noise_models;
    noise_models.accel_noise = make_imu_noise_model(acc_sigma, acc_bias_range, acc_bias_rw, 1234);
    noise_models.gyro_noise = make_imu_noise_model(gyr_sigma, gyr_bias_range, gyr_bias_rw, 5678);
    noise_models.mag_noise = make_mag_noise_model(mag_sigma_uT, 2.0f, 0.01f, 0.015f, 0.010f, 1.0f, 9012);

    const Vector3f sigma_a_init(2.8f * acc_sigma, 2.8f * acc_sigma, 2.8f * acc_sigma);
    const Vector3f sigma_g(2.0f * gyr_sigma, 2.0f * gyr_sigma, 2.0f * gyr_sigma);
    const float sigma_m_uT = 1.2f * mag_sigma_uT;
    const Vector3f sigma_m(sigma_m_uT, sigma_m_uT, sigma_m_uT);
    const Vector3f mag_world_a = MagSim_WMM::mag_world_aero();

    AdapterT adapter(with_mag, sigma_a_init, sigma_g, sigma_m, mag_world_a);
    W3dSimulationOptions options;
    options.dt = dt;
    options.with_mag = with_mag;
    options.add_noise = add_noise;
    options.mag_odr_hz = mag_odr_hz;
    options.temperature_c = 35.0f;

    W3dSimulationRunner runner(options, std::move(noise_models), adapter);
    return runner.run(filename);
}
