#include <filesystem>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

/*
    Copyright (c) 2025-2026  Mikhail Grushinskiy
*/

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "W3dSimCommon.h"
#include "AdaptiveVerticalPIIMahony.h"
#include "FrameConversions.h"
#include "Mahony_AHRS.h"

using Eigen::Vector3f;

bool add_noise = true;

static constexpr W3dFailureLimits FAIL_LIMITS{
    .err_limit_percent_z_jonswap = 17.9f,
    .err_limit_percent_z_pmstokes = 16.0f,
    .err_limit_yaw_deg = 5.06f,
};

class FusionAdapterAdaptivePIIMahony final : public IW3dFusionAdapter {
public:
    using HeaveFilter = marine_obs::AdaptiveVerticalPIIMahony<float, true, TrackerType::ARANOVSKIY>;

    FusionAdapterAdaptivePIIMahony(bool with_mag,
                                   const Vector3f& sigma_a_init,
                                   const Vector3f& sigma_g,
                                   const Vector3f& sigma_m,
                                   const Vector3f& mag_world_a)
        : with_mag_(with_mag),
          filter_(make_config_(with_mag, sigma_a_init, sigma_g, sigma_m, mag_world_a)),
          mag_declination_deg_(std::atan2(mag_world_a.y(), mag_world_a.x()) * 180.0f / float(M_PI))
    {
    }

    void updateMag(const Vector3f& mag_body_ned) override {
        // Runner supplies body-frame NED.
        last_mag_body_ned_ = mag_body_ned;
        have_mag_ = true;
    }

    void update(float dt,
                const Vector3f& gyr_meas_ned,
                const Vector3f& acc_meas_ned,
                float temperature_c) override
    {
        (void)temperature_c;

        // Runner-facing convention: body NED = (North, East, Down)
        // Mahony here is used in body NWU = (North, West, Up)
        // so that:
        //   - gravity is +Z at rest
        //   - horizontal north is +X for Mahony's mag update
        const Vector3f gyr_body_m = ned_to_mahony_body_(gyr_meas_ned);
        const Vector3f acc_body_m = ned_to_mahony_body_(acc_meas_ned);

        if (with_mag_ && have_mag_) {
            const Vector3f mag_body_m = ned_to_mahony_body_(last_mag_body_ned_);

            filter_.updateIMUMag(
                gyr_body_m.x(), gyr_body_m.y(), gyr_body_m.z(),
                acc_body_m.x(), acc_body_m.y(), acc_body_m.z(),
                mag_body_m.x(), mag_body_m.y(), mag_body_m.z(),
                dt
            );
        } else {
            filter_.updateIMU(
                gyr_body_m.x(), gyr_body_m.y(), gyr_body_m.z(),
                acc_body_m.x(), acc_body_m.y(), acc_body_m.z(),
                dt
            );
        }
    }

    FilterSnapshot snapshot() const override {
        FilterSnapshot s;

        const auto hs = filter_.snapshot();
        const auto& obs = hs.core.observer;

        s.disp_est_zu = Vector3f(0.0f, 0.0f, filter_.displacement());
        s.vel_est_zu  = Vector3f(0.0f, 0.0f, filter_.velocity());
        s.acc_est_zu  = Vector3f(0.0f, 0.0f, filter_.accelFiltered());

        // Mahony internal Euler here is in its NWU convention.
        // Convert back to the sim's nautical convention for RMS comparison.
        //
        // Derived mapping:
        //   roll_nautical  =  pitch_mahony
        //   pitch_nautical = -roll_mahony
        //   yaw_nautical   = -yaw_mahony (+ declination when mag is used)
        const float roll_m_deg  = filter_.rollDeg();
        const float pitch_m_deg = filter_.pitchDeg();
        const float yaw_m_deg   = filter_.yawDeg();

        const float roll_sim_deg  = -pitch_m_deg;
        const float pitch_sim_deg = roll_m_deg;
        const float yaw_sim_deg   = wrapDeg(-yaw_m_deg + (with_mag_ ? mag_declination_deg_ : 0.0f));

        s.euler_nautical_deg = Vector3f(roll_sim_deg,
                                        pitch_sim_deg,
                                        yaw_sim_deg);

        s.acc_bias_est_ned    = Vector3f::Zero();
        s.gyro_bias_est_ned   = Vector3f::Zero();
        s.mag_bias_est_ned_uT = Vector3f::Zero();

        const float r_active      = obs.r;
        const float tau_a_active  = obs.tau_a;
        const float tau_d_active  = obs.tau_d;
        const float kb_active     = obs.kb;

        const float sigma_raw     = hs.core.accel_sigma;
        const float sigma_used    = obs.sigma_a_filt;

        const float f_raw_hz      = hs.core.accel_freq_hz;
        const float f_used_hz     = (obs.f_disp_filt_hz > 1e-6f) ? obs.f_disp_filt_hz : f_raw_hz;
        const float omega_used    = (f_used_hz > 1e-6f) ? (2.0f * float(M_PI) * f_used_hz) : NAN;

        s.tau_target     = tau_d_active;
        s.sigma_target   = sigma_raw;
        s.tuning_target  = kb_active;

        s.tau_applied    = tau_a_active;
        s.sigma_applied  = sigma_used;
        s.tuning_applied = r_active;

        s.freq_hz = f_used_hz;
        s.period_sec = (f_used_hz > 1e-6f) ? (1.0f / f_used_hz) : NAN;
        s.accel_variance = hs.core.accel_var;

        if (std::isfinite(omega_used) && omega_used > 1e-6f &&
            std::isfinite(sigma_used) && sigma_used >= 0.0f) {
            s.displacement_scale_m = sigma_used / (omega_used * omega_used);
            s.velocity_scale_mps   = sigma_used / omega_used;
        } else {
            s.displacement_scale_m = NAN;
            s.velocity_scale_mps   = NAN;
        }

        s.direction.phase = NAN;
        s.direction.direction_deg = NAN;
        s.direction.direction_deg_generator_signed = NAN;
        s.direction.uncertainty_deg = NAN;
        s.direction.confidence = NAN;
        s.direction.amplitude = NAN;
        s.direction.direction_vec = Eigen::Vector2f::Zero();
        s.direction.filtered_signal = Eigen::Vector2f::Zero();
        s.direction.sign = UNCERTAIN;
        s.direction.sign_num = 0;

        return s;
    }

private:
    static Vector3f ned_to_mahony_body_(const Vector3f& v_ned) {
        // body NED (North, East, Down) -> body NWU (North, West, Up)
        return Vector3f(v_ned.x(), -v_ned.y(), -v_ned.z());
    }
        
    static HeaveFilter::Config make_config_(bool with_mag,
                                            const Vector3f& sigma_a_init,
                                            const Vector3f& sigma_g,
                                            const Vector3f& sigma_m,
                                            const Vector3f& mag_world_a)
    {
        (void)sigma_a_init;
        (void)sigma_g;
        (void)sigma_m;
        (void)mag_world_a;
    
        HeaveFilter::Config cfg{};
    
        // Base observer
        cfg.core.observer.r          = 0.150f;
        cfg.core.observer.tau_a      = 0.68f;
        cfg.core.observer.tau_d      = 49.0f;
        cfg.core.observer.kb         = 2.5e-5f;
        cfg.core.observer.lambda_b   = 3.0e-3f;
        cfg.core.observer.bias_limit = 0.12f;
    
        cfg.core.observer.a_f_limit = 50.0f;
        cfg.core.observer.v_limit   = 50.0f;
        cfg.core.observer.p_limit   = 20.0f;
        cfg.core.observer.S_limit   = 200.0f;
        cfg.core.observer.d_limit   = 20.0f;
    
        // Adaptation
        cfg.core.adaptation.enabled = true;
        cfg.core.adaptation.min_confidence = 0.22f;
    
        cfg.core.adaptation.f_disp_ref_hz    = 0.12f;
        cfg.core.adaptation.sigma_a_ref      = 0.95f;
        cfg.core.adaptation.input_smooth_tau = 4.5f;
        cfg.core.adaptation.param_smooth_tau = 7.5f;
    
        cfg.core.adaptation.r_freq_exp   = 0.28f;
        cfg.core.adaptation.r_sigma_exp  = 0.02f;
    
        cfg.core.adaptation.tau_a_freq_exp   = -0.40f;
        cfg.core.adaptation.tau_a_sigma_exp  = -0.03f;
    
        cfg.core.adaptation.tau_d_freq_exp   = -0.03f;
        cfg.core.adaptation.tau_d_sigma_exp  = -0.01f;
    
        cfg.core.adaptation.kb_freq_exp   = 0.02f;
        cfg.core.adaptation.kb_sigma_exp  = 0.08f;
    
        cfg.core.adaptation.r_min = 0.145f;
        cfg.core.adaptation.r_max = 0.225f;
    
        cfg.core.adaptation.tau_a_min = 0.50f;
        cfg.core.adaptation.tau_a_max = 0.90f;
    
        cfg.core.adaptation.tau_d_min = 44.0f;
        cfg.core.adaptation.tau_d_max = 58.0f;
    
        cfg.core.adaptation.kb_min = 5e-6f;
        cfg.core.adaptation.kb_max = 6e-5f;
    
        cfg.core.auto_schedule_from_accel_freq = true;
        cfg.core.auto_schedule_period_s = 0.50f;
        cfg.core.force_enable_adaptation_when_auto_schedule = true;
        cfg.core.fallback_confidence_floor = 0.52f;
        cfg.core.fallback_confidence_when_locked = 0.82f;
        cfg.core.coarse_schedule_blend = 0.48f;
        cfg.core.coarse_schedule_confidence_floor = 0.62f;
    
        cfg.core.accel_freq_tracker =
            marine_obs::detail::make_default_tracker_config<
                std::remove_cvref_t<decltype(cfg.core.accel_freq_tracker)>, float>();
    
        // Mahony base gains
        cfg.mahony_twoKp = 0.45f;
        cfg.mahony_twoKi = 0.015f;
        cfg.gravity_mps2 = g_std;
        cfg.use_mag = with_mag;
    
        // Mahony sea-state scheduling
        cfg.adapt_mahony_gains = true;
        cfg.mahony_twoKp_calm  = 0.90f;
        cfg.mahony_twoKp_rough = 0.35f;
        cfg.mahony_twoKi_calm  = 0.025f;
        cfg.mahony_twoKi_rough = 0.010f;
        cfg.mahony_sigma_ref = 0.18f;
        cfg.mahony_norm_err_ref = 0.08f;
        cfg.mahony_gain_smooth_tau_s = 2.0f;
        cfg.mahony_acc_trust_min = 0.05f;
    
        return cfg;
    }

private:
    bool with_mag_ = true;
    bool have_mag_ = false;

    Vector3f last_mag_body_ned_ = Vector3f::Zero();
    HeaveFilter filter_;
    float mag_declination_deg_ = 0.0f;
};

static void print_vertical_only_summary(const W3dSimulationRunResult& result, float dt)
{
    constexpr float RMS_WINDOW_SEC = 60.0f;
    const int N_last = static_cast<int>(RMS_WINDOW_SEC / dt);
    if (result.errs_z.size() <= static_cast<size_t>(N_last)) return;

    const size_t start = result.errs_z.size() - N_last;

    RMSReport rms_z, rms_roll, rms_pitch, rms_yaw;
    for (size_t i = start; i < result.errs_z.size(); ++i) {
        rms_z.add(result.errs_z[i]);
        rms_roll.add(result.errs_roll[i]);
        rms_pitch.add(result.errs_pitch[i]);
        rms_yaw.add(result.errs_yaw[i]);
    }

    const float z_rms = rms_z.rms();
    const float z_pct = 100.0f * z_rms / result.wave_params.height;

    std::vector<float> vf(result.freq_hist.begin() + start, result.freq_hist.end());

    std::cout << "=== Last 60 s VERTICAL-ONLY summary for " << result.output_name << " ===\n";
    std::cout << "Z RMS (m): " << z_rms << "\n";
    std::cout << "Z RMS (%Hs): " << z_pct << "% (Hs=" << result.wave_params.height << ")\n";
    std::cout << "Angles RMS (deg): Roll=" << rms_roll.rms()
              << " Pitch=" << rms_pitch.rms()
              << " Yaw=" << rms_yaw.rms() << "\n";

    std::cout << "f_used_hz: mean=" << mean_vec(vf)
              << " median=" << median_vec(vf)
              << " p05=" << percentile_vec(vf, 0.05)
              << " p95=" << percentile_vec(vf, 0.95) << "\n";

    std::cout << "active r=" << result.final_tuning_applied
              << ", active tau_a=" << result.final_tau_applied
              << ", active tau_d=" << result.final_tau_target
              << ", active kb=" << result.final_tuning_target << "\n";

    std::cout << "raw sigma_a=" << result.final_sigma_target
              << ", used sigma_a=" << result.final_sigma_applied
              << ", raw accel_var=" << result.final_accel_variance << "\n";

    std::cout << "===========================================================\n\n";
}

static void fail_if_vertical_quality_gates_breached(const W3dSimulationRunResult& result, float dt)
{
    constexpr float RMS_WINDOW_SEC = 60.0f;
    const int N_last = static_cast<int>(RMS_WINDOW_SEC / dt);
    if (result.errs_z.size() <= static_cast<size_t>(N_last)) return;

    const size_t start = result.errs_z.size() - N_last;

    RMSReport rms_z, rms_yaw;
    for (size_t i = start; i < result.errs_z.size(); ++i) {
        rms_z.add(result.errs_z[i]);
        rms_yaw.add(result.errs_yaw[i]);
    }

    const float z_pct = 100.0f * rms_z.rms() / result.wave_params.height;
    const float z_limit = (result.wave_type == WaveType::JONSWAP)
        ? FAIL_LIMITS.err_limit_percent_z_jonswap
        : FAIL_LIMITS.err_limit_percent_z_pmstokes;
    if (z_pct > z_limit) {
        std::cerr << "ERROR: Z RMS above limit (" << z_pct << "% > " << z_limit
                  << "%). Failing.\n";
        std::exit(EXIT_FAILURE);
    }

    if (result.with_mag && rms_yaw.rms() > FAIL_LIMITS.err_limit_yaw_deg) {
        std::cerr << "ERROR: Yaw RMS above limit (" << rms_yaw.rms() << " deg > "
                  << FAIL_LIMITS.err_limit_yaw_deg << " deg). Failing.\n";
        std::exit(EXIT_FAILURE);
    }
}

static std::optional<W3dSimulationRunResult>
process_wave_file_for_adaptive_pii_mahony(const std::string& filename,
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
    noise_models.gyro_noise  = make_imu_noise_model(gyr_sigma, gyr_bias_range, gyr_bias_rw, 5678);
    noise_models.mag_noise   = make_mag_noise_model(mag_sigma_uT, 2.0f, 0.01f,
                                                    0.015f, 0.010f, 1.0f, 9012);

    const Vector3f sigma_a_init(2.8f * acc_sigma, 2.8f * acc_sigma, 2.8f * acc_sigma);
    const Vector3f sigma_g(2.0f * gyr_sigma, 2.0f * gyr_sigma, 2.0f * gyr_sigma);
    const float sigma_m_uT = 1.2f * mag_sigma_uT;
    const Vector3f sigma_m(sigma_m_uT, sigma_m_uT, sigma_m_uT);
    const Vector3f mag_world_a = MagSim_WMM::mag_world_aero();

    FusionAdapterAdaptivePIIMahony adapter(with_mag, sigma_a_init, sigma_g, sigma_m, mag_world_a);

    W3dSimulationOptions options;
    options.dt = dt;
    options.with_mag = with_mag;
    options.add_noise = add_noise;
    options.mag_odr_hz = mag_odr_hz;
    options.temperature_c = 35.0f;
    options.output_suffix_with_mag = "_nonkalman_fusion";
    options.output_suffix_no_mag   = "_nonkalman_fusion_nomag";

    W3dSimulationRunner runner(options, std::move(noise_models), adapter);
    return runner.run(filename);
}

int main(int argc, char* argv[])
{
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

    std::cout << "AdaptiveVerticalPIIMahony simulation starting"
              << " with_mag=" << (with_mag ? "true" : "false")
              << ", noise=" << (add_noise ? "true" : "false")
              << "\n";

    const auto files = collect_wave_data_files(".");

    for (const auto& fname : files) {
        auto result = process_wave_file_for_adaptive_pii_mahony(
            fname,
            dt,
            with_mag,
            add_noise,
            25.0f
        );
        if (!result) continue;

        print_vertical_only_summary(*result, dt);
        fail_if_vertical_quality_gates_breached(*result, dt);
    }

    return 0;
}
