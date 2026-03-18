#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

/*
    Copyright (c) 2025-2026  Mikhail Grushinskiy
*/

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ZERO_CROSSINGS_SCALE          1.0f
#define ZERO_CROSSINGS_DEBOUNCE_TIME  0.12f
#define ZERO_CROSSINGS_STEEPNESS_TIME 0.21f

#define FREQ_GUESS 0.3f

const float g_std = 9.80665f;

const float FAIL_ERR_LIMIT_PERCENT_Z_JONSWAP = 9.97f;
const float FAIL_ERR_LIMIT_PERCENT_Z_PMSTOKES = 9.9f;
const float FAIL_ERR_LIMIT_YAW_DEG = 3.98f;
const float FAIL_ERR_LIMIT_PERCENT_3D_JONSWAP = 50.0f;
const float FAIL_ERR_LIMIT_PERCENT_3D_PMSTOKES = 50.0f;
const float FAIL_ACC_Z_BIAS_PERCENT = 28.0f;
const float FAIL_ERR_LIMIT_BIAS_3D_PERCENT = 250.0f;

constexpr float RMS_WINDOW_SEC = 60.0f;

#include "W3dSimCommon.h"
#include "SeaStateFusionFilter_4.h"

using Eigen::Vector2f;
using Eigen::Vector3f;

template<typename T>
static T mean_vec(const std::vector<T>& v){
    if (v.empty()) return T(NAN);
    T s = 0;
    for (const auto& x : v) s += x;
    return s / T(v.size());
}

template<typename T>
static T median_vec(std::vector<T> v){
    if (v.empty()) return T(NAN);
    size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    if (n % 2) return v[n / 2];
    auto lo = *std::max_element(v.begin(), v.begin() + n / 2);
    auto hi = v[n / 2];
    return (lo + hi) / T(2);
}

template<typename T>
static T percentile_vec(std::vector<T> v, double p01){
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

static inline float deg_to_rad(float d){ return d * float(M_PI / 180.0); }
static inline float rad_to_deg(float r){ return r * float(180.0 / M_PI); }

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

struct CircStats {
    float mean_deg = NAN;
    float std_deg = NAN;
};

static CircStats circular_stats_180(const std::vector<float>& degs){
    CircStats cs;
    if (degs.empty()) return cs;

    double C = 0, S = 0;
    for (float d : degs){
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

bool add_noise = true;
bool attitude_only = false;

class FusionAdapter4 final : public IW3dFusionAdapter {
public:
    FusionAdapter4(bool with_mag,
                   const Vector3f& sigma_a_init,
                   const Vector3f& sigma_g,
                   const Vector3f& sigma_m,
                   const Vector3f& mag_world_a)
    {
        cfg_.with_mag = with_mag;
        cfg_.sigma_a = sigma_a_init;
        cfg_.sigma_g = sigma_g;
        cfg_.sigma_m = sigma_m;
        cfg_.mag_delay_sec = MAG_DELAY_SEC;
        cfg_.use_fixed_mag_world_ref = false;
        cfg_.mag_world_ref = mag_world_a;
        cfg_.freeze_acc_bias_until_live = true;
        cfg_.Racc_warmup = 0.5f;

        fusion_.begin(cfg_);
        auto& filter = fusion_.raw();
        if (attitude_only) {
            filter.enableLinearBlock(false);
            filter.mekf().set_initial_acc_bias(Vector3f::Zero());
            filter.mekf().set_initial_acc_bias_std(0.0f);
            filter.mekf().set_Q_bacc_rw(Vector3f::Zero());
            filter.mekf().set_Racc_std(Vector3f::Constant(0.4f));
        } else {
            filter.enableLinearBlock(true);
            filter.enableTuner(true);
            filter.enableClamp(true);
        }
    }

    void updateMag(const Vector3f& mag_body_ned) override { fusion_.updateMag(mag_body_ned); }

    void update(float dt,
                const Vector3f& gyr_meas_ned,
                const Vector3f& acc_meas_ned,
                float temperature_c) override {
        fusion_.update(dt, gyr_meas_ned, acc_meas_ned, temperature_c);
    }

    FilterSnapshot snapshot() const override {
        const auto& filter = fusion_.raw();
        const auto& d = filter.dir();

        FilterSnapshot s;
        s.disp_est_zu = ned_to_zu(filter.mekf().get_position());
        s.vel_est_zu = ned_to_zu(filter.mekf().get_velocity());
        s.acc_est_zu = ned_to_zu(filter.mekf().get_world_accel());
        s.euler_nautical_deg = filter.getEulerNautical();
        s.acc_bias_est_ned = filter.mekf().get_acc_bias();
        s.gyro_bias_est_ned = filter.mekf().gyroscope_bias();
        s.mag_bias_est_ned_uT = get_mag_bias_est_uT(filter.mekf());
        s.tau_target = filter.getTauTarget();
        s.sigma_target = filter.getSigmaTarget();
        s.tuning_target = p0_s_from_sigma_tau(s.sigma_target, s.tau_target);
        s.tau_applied = filter.getTauApplied();
        s.sigma_applied = filter.getSigmaApplied();
        s.tuning_applied = p0_s_from_sigma_tau(s.sigma_applied, s.tau_applied);
        s.freq_hz = filter.getFreqHz();
        s.period_sec = filter.getPeriodSec();
        s.accel_variance = filter.getAccelVariance();
        s.displacement_scale_m = filter.getDisplacementScale();
        s.velocity_scale_mps = filter.getVerticalSpeedEnvelopeMps(true);
        s.direction.phase = d.getPhase();
        s.direction.direction_deg = d.getDirectionDegrees();
        s.direction.direction_deg_generator_signed = dirDegGeneratorSignedFromVec(d.getDirection());
        s.direction.uncertainty_deg = d.getDirectionUncertaintyDegrees();
        s.direction.confidence = d.getLastStableConfidence();
        s.direction.amplitude = d.getAmplitude();
        s.direction.direction_vec = d.getDirection();
        s.direction.filtered_signal = d.getFilteredSignal();
        constexpr float CONF_THRESH = 20.0f;
        constexpr float AMP_THRESH = 0.08f;
        if (s.direction.confidence > CONF_THRESH && s.direction.amplitude > AMP_THRESH) {
            s.direction.sign = filter.getDirSignState();
            s.direction.sign_num = (s.direction.sign == FORWARD) ? 1 : (s.direction.sign == BACKWARD ? -1 : 0);
        }
        return s;
    }

private:
    using Fusion = SeaStateFusion_4<TrackerType::KALMANF>;
    mutable Fusion fusion_;
    Fusion::Config cfg_{};
};

static void print_summary_and_fail_if_needed(const W3dSimulationRunResult& result, float dt)
{
    const int N_last = static_cast<int>(RMS_WINDOW_SEC / dt);
    if (result.errs_z.size() <= static_cast<size_t>(N_last)) return;

    const size_t start = result.errs_z.size() - N_last;
    RMSReport rms_x, rms_y, rms_z, rms_roll, rms_pitch, rms_yaw;
    RMSReport rms_accb_x, rms_accb_y, rms_accb_z;
    RMSReport rms_gyrb_x, rms_gyrb_y, rms_gyrb_z;
    RMSReport rms_magb_x, rms_magb_y, rms_magb_z;

    float acc_true_max_x = 0.f, acc_true_max_y = 0.f, acc_true_max_z = 0.f, acc_true_max_3d = 0.f;
    float gyr_true_max_x = 0.f, gyr_true_max_y = 0.f, gyr_true_max_z = 0.f, gyr_true_max_3d = 0.f;
    float mag_true_max_x = 0.f, mag_true_max_y = 0.f, mag_true_max_z = 0.f, mag_true_max_3d = 0.f;
    float disp_true_max_3d = 0.f;

    for (size_t i = start; i < result.errs_z.size(); ++i) {
        rms_x.add(result.errs_x[i]);
        rms_y.add(result.errs_y[i]);
        rms_z.add(result.errs_z[i]);
        rms_roll.add(result.errs_roll[i]);
        rms_pitch.add(result.errs_pitch[i]);
        rms_yaw.add(result.errs_yaw[i]);
        rms_accb_x.add(result.accb_err_x[i]);
        rms_accb_y.add(result.accb_err_y[i]);
        rms_accb_z.add(result.accb_err_z[i]);
        rms_gyrb_x.add(result.gyrb_err_x[i]);
        rms_gyrb_y.add(result.gyrb_err_y[i]);
        rms_gyrb_z.add(result.gyrb_err_z[i]);
        rms_magb_x.add(result.magb_err_x[i]);
        rms_magb_y.add(result.magb_err_y[i]);
        rms_magb_z.add(result.magb_err_z[i]);

        const float dx = result.ref_x[i];
        const float dy = result.ref_y[i];
        const float dz = result.ref_z[i];
        disp_true_max_3d = std::max(disp_true_max_3d, std::sqrt(dx * dx + dy * dy + dz * dz));

        const float ax = result.accb_true_x[i], ay = result.accb_true_y[i], az = result.accb_true_z[i];
        acc_true_max_x = std::max(acc_true_max_x, std::abs(ax));
        acc_true_max_y = std::max(acc_true_max_y, std::abs(ay));
        acc_true_max_z = std::max(acc_true_max_z, std::abs(az));
        acc_true_max_3d = std::max(acc_true_max_3d, std::sqrt(ax * ax + ay * ay + az * az));

        const float gx = result.gyrb_true_x[i], gy = result.gyrb_true_y[i], gz = result.gyrb_true_z[i];
        gyr_true_max_x = std::max(gyr_true_max_x, std::abs(gx));
        gyr_true_max_y = std::max(gyr_true_max_y, std::abs(gy));
        gyr_true_max_z = std::max(gyr_true_max_z, std::abs(gz));
        gyr_true_max_3d = std::max(gyr_true_max_3d, std::sqrt(gx * gx + gy * gy + gz * gz));

        const float mx = result.magb_true_x[i], my = result.magb_true_y[i], mz = result.magb_true_z[i];
        mag_true_max_x = std::max(mag_true_max_x, std::abs(mx));
        mag_true_max_y = std::max(mag_true_max_y, std::abs(my));
        mag_true_max_z = std::max(mag_true_max_z, std::abs(mz));
        mag_true_max_3d = std::max(mag_true_max_3d, std::sqrt(mx * mx + my * my + mz * mz));
    }

    const float x_rms = rms_x.rms(), y_rms = rms_y.rms(), z_rms = rms_z.rms();
    const float x_pct = 100.f * x_rms / result.wave_params.height;
    const float y_pct = 100.f * y_rms / result.wave_params.height;
    const float z_pct = 100.f * z_rms / result.wave_params.height;
    const float rms_3d_err = std::sqrt(x_rms * x_rms + y_rms * y_rms + z_rms * z_rms);
    const float pct_3d = (disp_true_max_3d > 1e-12f && std::isfinite(rms_3d_err))
        ? 100.f * rms_3d_err / disp_true_max_3d
        : NAN;

    std::cout << "=== Last 60 s RMS summary for " << result.output_name << " ===\n";
    std::cout << "XYZ RMS (m): X=" << x_rms << " Y=" << y_rms << " Z=" << z_rms << "\n";
    std::cout << "XYZ RMS (%Hs): X=" << x_pct << "% Y=" << y_pct << "% Z=" << z_pct
              << "% (Hs=" << result.wave_params.height << ")\n";
    std::cout << "3D RMS (m): " << rms_3d_err
              << " (3D % of max |disp_ref|_3D = " << pct_3d
              << "%, max |disp_ref|_3D = " << disp_true_max_3d << " m)\n";
    std::cout << "Angles RMS (deg): Roll=" << rms_roll.rms()
              << " Pitch=" << rms_pitch.rms()
              << " Yaw=" << rms_yaw.rms() << "\n";

    auto vec_rms = [](float rx, float ry, float rz) { return std::sqrt(rx * rx + ry * ry + rz * rz); };
    const float accb_rx = rms_accb_x.rms(), accb_ry = rms_accb_y.rms(), accb_rz = rms_accb_z.rms();
    const float gyrb_rx = rms_gyrb_x.rms(), gyrb_ry = rms_gyrb_y.rms(), gyrb_rz = rms_gyrb_z.rms();
    const float magb_rx = rms_magb_x.rms(), magb_ry = rms_magb_y.rms(), magb_rz = rms_magb_z.rms();
    const float accb_r3 = vec_rms(accb_rx, accb_ry, accb_rz);
    const float gyrb_r3 = vec_rms(gyrb_rx, gyrb_ry, gyrb_rz);
    const float magb_r3 = vec_rms(magb_rx, magb_ry, magb_rz);

    std::cout << "Bias error RMS (acc, m/s^2): X=" << accb_rx << " Y=" << accb_ry << " Z=" << accb_rz
              << " |3D|=" << accb_r3 << "\n";
    std::cout << "Bias error RMS (gyro, rad/s): X=" << gyrb_rx << " Y=" << gyrb_ry << " Z=" << gyrb_rz
              << " |3D|=" << gyrb_r3 << "\n";
    const float rad2deg = 180.0f / float(M_PI);
    std::cout << "Bias error RMS (gyro, deg/s): X=" << (gyrb_rx * rad2deg)
              << " Y=" << (gyrb_ry * rad2deg)
              << " Z=" << (gyrb_rz * rad2deg)
              << " |3D|=" << (gyrb_r3 * rad2deg) << "\n";
#ifdef DETAILED_SUMMARY
    std::cout << "Bias error RMS (mag, uT): X=" << magb_rx << " Y=" << magb_ry << " Z=" << magb_rz
              << " |3D|=" << magb_r3 << "\n";
#endif

    auto pct_of_max = [](float rms, float maxv) -> float {
        return (maxv > 1e-12f && std::isfinite(rms)) ? (100.f * rms / maxv) : NAN;
    };

    std::cout << "Max TRUE bias in window (acc, m/s^2): X=" << acc_true_max_x << " Y=" << acc_true_max_y
              << " Z=" << acc_true_max_z << " |3D|=" << acc_true_max_3d << "\n";
    std::cout << "Max TRUE bias in window (gyro, rad/s): X=" << gyr_true_max_x << " Y=" << gyr_true_max_y
              << " Z=" << gyr_true_max_z << " |3D|=" << gyr_true_max_3d << "\n";
#ifdef DETAILED_SUMMARY
    std::cout << "Max TRUE bias in window (mag, uT): X=" << mag_true_max_x << " Y=" << mag_true_max_y
              << " Z=" << mag_true_max_z << " |3D|=" << mag_true_max_3d << "\n";
#endif

    const float accb_r3_pct = pct_of_max(accb_r3, acc_true_max_3d);
    const float gyrb_r3_pct = pct_of_max(gyrb_r3, gyr_true_max_3d);
    const float magb_r3_pct = pct_of_max(magb_r3, mag_true_max_3d);
    std::cout << "Bias error RMS (% of max TRUE bias) (acc): X=" << pct_of_max(accb_rx, acc_true_max_x)
              << "% Y=" << pct_of_max(accb_ry, acc_true_max_y)
              << "% Z=" << pct_of_max(accb_rz, acc_true_max_z)
              << "% |3D|=" << accb_r3_pct << "%\n";
    std::cout << "Bias error RMS (% of max TRUE bias) (gyro): X=" << pct_of_max(gyrb_rx, gyr_true_max_x)
              << "% Y=" << pct_of_max(gyrb_ry, gyr_true_max_y)
              << "% Z=" << pct_of_max(gyrb_rz, gyr_true_max_z)
              << "% |3D|=" << gyrb_r3_pct << "%\n";
#ifdef DETAILED_SUMMARY
    std::cout << "Bias error RMS (% of max TRUE bias) (mag): X=" << pct_of_max(magb_rx, mag_true_max_x)
              << "% Y=" << pct_of_max(magb_ry, mag_true_max_y)
              << "% Z=" << pct_of_max(magb_rz, mag_true_max_z)
              << "% |3D|=" << magb_r3_pct << "%\n";
#endif

    std::cout << "tau_target=" << result.final_tau_target
              << ", sigma_target=" << result.final_sigma_target
              << ", p0_S_target=" << result.final_tuning_target << "\n";
    std::cout << "tau_applied=" << result.final_tau_applied
              << ", sigma_applied=" << result.final_sigma_applied
              << ", p0_S_applied=" << result.final_tuning_applied << "\n";
    std::cout << "f_hz=" << result.final_freq_hz
              << ", Tp_tuner=" << result.final_period_sec
              << ", accel_var=" << result.final_accel_variance << "\n";

    if (start < result.dir_deg_hist.size()) {
        const size_t i0 = start;
        const size_t i1 = result.errs_z.size();
        std::vector<float> vf(result.freq_hist.begin() + i0, result.freq_hist.begin() + i1);
        std::vector<float> vd(result.dir_deg_hist.begin() + i0, result.dir_deg_hist.begin() + i1);
        std::vector<float> vu(result.dir_unc_hist.begin() + i0, result.dir_unc_hist.begin() + i1);
        std::vector<float> vc(result.dir_conf_hist.begin() + i0, result.dir_conf_hist.begin() + i1);
        vd.erase(std::remove_if(vd.begin(), vd.end(), [](float a){ return !std::isfinite(a); }), vd.end());
        auto cs = circular_stats_180(vd);

        int nToward = 0, nAway = 0, nUnc = 0;
        size_t good = 0;
        constexpr float CONF_THRESH = 20.0f;
        constexpr float AMP_THRESH = 0.08f;
        for (size_t k = i0; k < i1; ++k) {
            const int s = result.dir_sign_num_hist[k];
            if (s > 0) ++nToward;
            else if (s < 0) ++nAway;
            else ++nUnc;
            if (result.dir_conf_hist[k] > CONF_THRESH && result.dir_amp_hist[k] > AMP_THRESH) ++good;
        }
        const int nWin = int(i1 - i0);
        auto pct = [&](int n){ return (nWin > 0) ? (100.0 * double(n) / double(nWin)) : 0.0; };

        std::cout << "=== Direction Report (last 60 s only) for " << result.output_name << " ===\n";
        std::cout << "window_s: " << (float(i1 - i0) * dt) << " samples: " << (i1 - i0) << "\n";
        std::cout << "freq_hz: mean=" << mean_vec(vf)
                  << " median=" << median_vec(vf)
                  << " p05=" << percentile_vec(vf, 0.05)
                  << " p95=" << percentile_vec(vf, 0.95) << "\n";
        std::cout << "dir_deg_gen ([-90,90], 0=+Y CW): mean_circ=" << cs.mean_deg
                  << " circ_std≈" << cs.std_deg << " deg\n";
        std::cout << "uncert_deg: mean=" << mean_vec(vu)
                  << " median=" << median_vec(vu)
                  << " p95=" << percentile_vec(vu, 0.95) << "\n";
        std::cout << "confidence: mean=" << mean_vec(vc)
                  << " >" << CONF_THRESH << " count=" << good
                  << " (" << (100.0 * double(good) / double(i1 - i0)) << "%)\n";
        std::cout << "sign: TOWARD=" << nToward << " (" << pct(nToward) << "%)"
                  << " AWAY=" << nAway << " (" << pct(nAway) << "%)"
                  << " UNCERTAIN=" << nUnc << " (" << pct(nUnc) << "%)\n";
        std::cout << "=============================================\n\n";
    }

    const float limit_z = (result.wave_type == WaveType::JONSWAP) ? FAIL_ERR_LIMIT_PERCENT_Z_JONSWAP : FAIL_ERR_LIMIT_PERCENT_Z_PMSTOKES;
    const float limit_3d = (result.wave_type == WaveType::JONSWAP) ? FAIL_ERR_LIMIT_PERCENT_3D_JONSWAP : FAIL_ERR_LIMIT_PERCENT_3D_PMSTOKES;
    auto fail_if = [&](const char* label, float pct, float limit) {
        if (pct > limit) {
            std::cerr << "ERROR: " << label << " RMS above limit (" << pct << "% > " << limit << "%). Failing.\n";
            std::exit(EXIT_FAILURE);
        }
    };

    fail_if("Z", z_pct, limit_z);
    fail_if("3D", pct_3d, limit_3d);

    if (rms_yaw.rms() > FAIL_ERR_LIMIT_YAW_DEG) {
        std::cerr << "ERROR: Yaw RMS above limit (" << rms_yaw.rms() << " deg > " << FAIL_ERR_LIMIT_YAW_DEG << " deg). Failing.\n";
        std::exit(EXIT_FAILURE);
    }

    const float accb_z_pct = pct_of_max(accb_rz, acc_true_max_z);
    if (std::isfinite(accb_z_pct) && accb_z_pct > FAIL_ACC_Z_BIAS_PERCENT) {
        std::cerr << "ERROR: accel Z bias error RMS above limit ("
                  << accb_z_pct << "% > " << FAIL_ACC_Z_BIAS_PERCENT << "% of max TRUE Z bias). Failing.\n";
        std::exit(EXIT_FAILURE);
    }
    if (std::isfinite(accb_r3_pct) && accb_r3_pct > FAIL_ERR_LIMIT_BIAS_3D_PERCENT) {
        std::cerr << "ERROR: 3D accel bias error RMS above limit ("
                  << accb_r3_pct << "% > " << FAIL_ERR_LIMIT_BIAS_3D_PERCENT << "% of max TRUE bias). Failing.\n";
        std::exit(EXIT_FAILURE);
    }
    if (std::isfinite(gyrb_r3_pct) && gyrb_r3_pct > FAIL_ERR_LIMIT_BIAS_3D_PERCENT) {
        std::cerr << "ERROR: 3D gyro bias error RMS above limit ("
                  << gyrb_r3_pct << "% > " << FAIL_ERR_LIMIT_BIAS_3D_PERCENT << "% of max TRUE bias). Failing.\n";
        std::exit(EXIT_FAILURE);
    }
}

static void process_wave_file_for_tracker(const std::string& filename, float dt, bool with_mag)
{
    const float acc_sigma = 1.51e-3f * g_std;
    const float gyr_sigma = 0.00157f;
    const float acc_bias_range = 5e-3f * g_std;
    const float gyr_bias_range = 0.05f * float(M_PI / 180.0f);
    const float acc_bias_rw = 0.0005f;
    const float gyr_bias_rw = 0.00001f;
    constexpr float MAG_ODR_HZ = 25.0f;
    const float mag_sigma_uT = (MAG_ODR_HZ <= 20.0f) ? 0.30f : 0.60f;

    SimulationNoiseModels noise_models;
    noise_models.accel_noise = make_imu_noise_model(acc_sigma, acc_bias_range, acc_bias_rw, 1234);
    noise_models.gyro_noise = make_imu_noise_model(gyr_sigma, gyr_bias_range, gyr_bias_rw, 5678);
    noise_models.mag_noise = make_mag_noise_model(mag_sigma_uT, 2.0f, 0.01f, 0.015f, 0.010f, 1.0f, 9012);

    const Vector3f sigma_a_init(2.8f * acc_sigma, 2.8f * acc_sigma, 2.8f * acc_sigma);
    const Vector3f sigma_g(2.0f * gyr_sigma, 2.0f * gyr_sigma, 2.0f * gyr_sigma);
    const float sigma_m_uT = 1.2f * mag_sigma_uT;
    const Vector3f sigma_m(sigma_m_uT, sigma_m_uT, sigma_m_uT);
    const Vector3f mag_world_a = MagSim_WMM::mag_world_aero();

    FusionAdapter4 adapter(with_mag, sigma_a_init, sigma_g, sigma_m, mag_world_a);
    W3dSimulationOptions options;
    options.dt = dt;
    options.with_mag = with_mag;
    options.add_noise = add_noise;
    options.mag_odr_hz = MAG_ODR_HZ;
    options.temperature_c = 35.0f;

    W3dSimulationRunner runner(options, std::move(noise_models), adapter);
    auto result = runner.run(filename);
    if (!result) return;
    print_summary_and_fail_if_needed(*result, dt);
}

int main(int argc, char* argv[]) {
    const float dt = 1.0f / 200.0f;
    bool with_mag = true;
    add_noise = true;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--nomag") {
            with_mag = false;
        } else if (arg == "--no-noise") {
            add_noise = false;
        }
    }

    std::cout << "Simulation starting with_mag=" << (with_mag ? "true" : "false")
              << ", mag_delay=" << MAG_DELAY_SEC
              << " sec, noise=" << (add_noise ? "true" : "false")
              << "\n";

    std::vector<std::string> files;
    for (auto& entry : std::filesystem::directory_iterator(".")) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().string();
        if (fname.find("wave_data_") == std::string::npos) continue;
        if (auto kind = WaveFileNaming::parse_kind_only(fname); kind && *kind == FileKind::Data) {
            files.push_back(std::move(fname));
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto& fname : files) {
        process_wave_file_for_tracker(fname, dt, with_mag);
    }
    return 0;
}
