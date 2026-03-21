// w3d_sim_adaptive_pii_mahony.cpp

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

#include "W3dSimCommon.h"
#include "AdaptiveVerticalPIIMahony.h"
#include "FrameConversions.h"
#include "Mahony_AHRS.h"

using Eigen::Vector3f;

bool add_noise = true;

// Heave-only adapter:
// - accepts NED inputs from W3dSimulationRunner
// - converts them back to Z-up/body convention for the Mahony-based wrapper
// - outputs only Z displacement / velocity / vertical accel estimate
class FusionAdapterAdaptivePIIMahony final : public IW3dFusionAdapter {
public:
    using HeaveFilter = marine_obs::AdaptiveVerticalPIIMahony<float, true>;

    FusionAdapterAdaptivePIIMahony(bool with_mag,
                                   const Vector3f& sigma_a_init,
                                   const Vector3f& sigma_g,
                                   const Vector3f& sigma_m,
                                   const Vector3f& mag_world_a)
        : with_mag_(with_mag),
          filter_(make_config_(with_mag, sigma_a_init, sigma_g, sigma_m, mag_world_a))
    {
        (void)sigma_g;
        (void)sigma_m;
        (void)mag_world_a;
    }

    void updateMag(const Vector3f& mag_body_ned) override {
        // Runner gives NED; Mahony wrapper expects the Z-up convention
        last_mag_body_zu_ = ned_to_zu(mag_body_ned);
        have_mag_ = true;
    }

    void update(float dt,
                const Vector3f& gyr_meas_ned,
                const Vector3f& acc_meas_ned,
                float temperature_c) override
    {
        (void)temperature_c;

        // Convert back from runner's NED body convention to the Mahony wrapper's Z-up body convention
        const Vector3f gyr_body_zu = ned_to_zu(gyr_meas_ned);
        const Vector3f acc_body_zu = ned_to_zu(acc_meas_ned);

        if (with_mag_ && have_mag_) {
            filter_.updateIMUMag(
                gyr_body_zu.x(), gyr_body_zu.y(), gyr_body_zu.z(),
                acc_body_zu.x(), acc_body_zu.y(), acc_body_zu.z(),
                last_mag_body_zu_.x(), last_mag_body_zu_.y(), last_mag_body_zu_.z(),
                dt
            );
        } else {
            filter_.updateIMU(
                gyr_body_zu.x(), gyr_body_zu.y(), gyr_body_zu.z(),
                acc_body_zu.x(), acc_body_zu.y(), acc_body_zu.z(),
                dt
            );
        }
    }

    FilterSnapshot snapshot() const override {
        FilterSnapshot s;

        const auto hs = filter_.snapshot();
        const auto& obs = hs.core.observer;

        // Heave-only outputs in Z-up world
        s.disp_est_zu = Vector3f(0.0f, 0.0f, filter_.displacement());
        s.vel_est_zu  = Vector3f(0.0f, 0.0f, filter_.velocity());
        s.acc_est_zu  = Vector3f(0.0f, 0.0f, filter_.accelFiltered());

        // Keep Mahony Euler outputs as-is for diagnostics.
        s.euler_nautical_deg = Vector3f(filter_.rollDeg(),
                                        filter_.pitchDeg(),
                                        filter_.yawDeg());

        // This observer does not estimate full sensor biases.
        s.acc_bias_est_ned    = Vector3f::Zero();
        s.gyro_bias_est_ned   = Vector3f::Zero();
        s.mag_bias_est_ned_uT = Vector3f::Zero();

        // -----------------------------
        // Actual adaptation diagnostics
        // -----------------------------
        //
        // obs.r            = active repeated-pole rate actually used
        // obs.tau_a        = active accel LPF tau actually used
        // obs.tau_d        = active slow trend tau actually used
        // obs.kb           = active bias gain actually used
        //
        // hs.core.accel_sigma        = raw wrapper sigma estimate
        // obs.sigma_a_filt           = smoothed sigma actually used by observer adaptation
        // hs.core.accel_freq_hz      = raw tracker frequency
        // obs.f_disp_filt_hz         = smoothed frequency actually used by observer adaptation

        const float r_active      = obs.r;
        const float tau_a_active  = obs.tau_a;
        const float tau_d_active  = obs.tau_d;
        const float kb_active     = obs.kb;

        const float sigma_raw     = hs.core.accel_sigma;
        const float sigma_used    = obs.sigma_a_filt;

        const float f_raw_hz      = hs.core.accel_freq_hz;
        const float f_used_hz     = (obs.f_disp_filt_hz > 1e-6f) ? obs.f_disp_filt_hz : f_raw_hz;
        const float omega_used    = (f_used_hz > 1e-6f) ? (2.0f * float(M_PI) * f_used_hz) : NAN;

        // We reuse generic fields, but now they contain real observer parameters:
        //
        //   tau_target    -> active tau_d
        //   sigma_target  -> raw sigma estimate
        //   tuning_target -> active kb
        //
        //   tau_applied    -> active tau_a
        //   sigma_applied  -> smoothed sigma actually used
        //   tuning_applied -> active r
        s.tau_target     = tau_d_active;
        s.sigma_target   = sigma_raw;
        s.tuning_target  = kb_active;

        s.tau_applied    = tau_a_active;
        s.sigma_applied  = sigma_used;
        s.tuning_applied = r_active;

        // Keep freq_hz history as RAW tracker frequency so you can see tracker behavior.
        s.freq_hz = f_raw_hz;

        // Store USED scheduling period here.
        s.period_sec = (f_used_hz > 1e-6f) ? (1.0f / f_used_hz) : NAN;

        // Keep this as raw variance estimate from the wrapper.
        s.accel_variance = hs.core.accel_var;

        // Diagnostic scales based on actually used scheduled values.
        if (std::isfinite(omega_used) && omega_used > 1e-6f &&
            std::isfinite(sigma_used) && sigma_used >= 0.0f) {
            s.displacement_scale_m = sigma_used / (omega_used * omega_used);
            s.velocity_scale_mps   = sigma_used / omega_used;
        } else {
            s.displacement_scale_m = NAN;
            s.velocity_scale_mps   = NAN;
        }

        // Direction telemetry not available in this heave-only filter
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

        // ----------------
        // Core observer
        // ----------------
        cfg.core.observer.r         = 0.16f;
        cfg.core.observer.tau_a     = 0.60f;
        cfg.core.observer.tau_d     = 40.0f;
        cfg.core.observer.kb        = 1e-4f;
        cfg.core.observer.lambda_b  = 1e-2f;
        cfg.core.observer.bias_limit = 0.25f;

        // Safety limits
        cfg.core.observer.a_f_limit = 50.0f;
        cfg.core.observer.v_limit   = 50.0f;
        cfg.core.observer.p_limit   = 20.0f;
        cfg.core.observer.S_limit   = 200.0f;
        cfg.core.observer.d_limit   = 20.0f;

        // ----------------
        // Adaptation
        // ----------------
        cfg.core.adaptation.enabled = true;
        cfg.core.adaptation.min_confidence = 0.25f;

        cfg.core.adaptation.f_disp_ref_hz = 0.17f;
        cfg.core.adaptation.sigma_a_ref   = 0.30f;
        cfg.core.adaptation.input_smooth_tau = 5.0f;
        cfg.core.adaptation.param_smooth_tau = 10.0f;

        cfg.core.adaptation.r_freq_exp  = 0.50f;
        cfg.core.adaptation.r_sigma_exp = 0.50f;

        cfg.core.adaptation.tau_a_freq_exp  = -0.75f;
        cfg.core.adaptation.tau_a_sigma_exp = -0.50f;

        cfg.core.adaptation.tau_d_freq_exp  = 0.0f;
        cfg.core.adaptation.tau_d_sigma_exp = -0.50f;

        cfg.core.adaptation.kb_freq_exp  = 0.0f;
        cfg.core.adaptation.kb_sigma_exp = 1.00f;

        cfg.core.adaptation.r_min = 0.05f;
        cfg.core.adaptation.r_max = 0.30f;

        cfg.core.adaptation.tau_a_min = 0.10f;
        cfg.core.adaptation.tau_a_max = 2.00f;

        cfg.core.adaptation.tau_d_min = 5.0f;
        cfg.core.adaptation.tau_d_max = 120.0f;

        cfg.core.adaptation.kb_min = 0.0f;
        cfg.core.adaptation.kb_max = 1e-2f;

        // ----------------
        // Auto fallback scheduling from internal accel-frequency tracker
        // ----------------
        cfg.core.auto_schedule_from_accel_freq = true;
        cfg.core.auto_schedule_period_s = 0.25f;
        cfg.core.force_enable_adaptation_when_auto_schedule = true;

        // These are the important fixes for the "it does nothing" feeling.
        cfg.core.fallback_confidence_floor = 0.35f;
        cfg.core.fallback_confidence_when_locked = 0.70f;

        // ----------------
        // Internal acceleration-frequency tracker
        // ----------------
        cfg.core.accel_freq_tracker.f_min_hz = 0.04f;
        cfg.core.accel_freq_tracker.f_max_hz = 0.60f;
        cfg.core.accel_freq_tracker.f_init_hz = 0.17f;

        cfg.core.accel_freq_tracker.pre_hp_hz = 0.03f;
        cfg.core.accel_freq_tracker.pre_lp_hz = 0.80f;
        cfg.core.accel_freq_tracker.demod_lp_hz = 0.08f;

        cfg.core.accel_freq_tracker.loop_bandwidth_hz = 0.03f;
        cfg.core.accel_freq_tracker.loop_damping = 0.90f;

        cfg.core.accel_freq_tracker.output_smooth_tau_s = 1.5f;
        cfg.core.accel_freq_tracker.power_tau_s = 6.0f;
        cfg.core.accel_freq_tracker.confidence_tau_s = 3.0f;
        cfg.core.accel_freq_tracker.lock_rms_min = 0.005f;
        cfg.core.accel_freq_tracker.enable_coarse_assist = true;

        // ----------------
        // Mahony
        // ----------------
        cfg.mahony_twoKp = twoKpDef;
        cfg.mahony_twoKi = twoKiDef;
        cfg.gravity_mps2 = g_std;
        cfg.use_mag = with_mag;

        return cfg;
    }

private:
    bool with_mag_ = true;
    bool have_mag_ = false;
    Vector3f last_mag_body_zu_ = Vector3f::Zero();
    HeaveFilter filter_;
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

    const float f_used_last =
        (std::isfinite(result.final_period_sec) && result.final_period_sec > 1e-6f)
            ? (1.0f / result.final_period_sec)
            : NAN;

    std::cout << "=== Last 60 s VERTICAL-ONLY summary for " << result.output_name << " ===\n";
    std::cout << "Z RMS (m): " << z_rms << "\n";
    std::cout << "Z RMS (%Hs): " << z_pct << "% (Hs=" << result.wave_params.height << ")\n";
    std::cout << "Angles RMS (deg): Roll=" << rms_roll.rms()
              << " Pitch=" << rms_pitch.rms()
              << " Yaw=" << rms_yaw.rms() << "\n";

    std::cout << "f_raw_hz: mean=" << mean_vec(vf)
              << " median=" << median_vec(vf)
              << " p05=" << percentile_vec(vf, 0.05)
              << " p95=" << percentile_vec(vf, 0.95) << "\n";

    std::cout << "last raw f_hz=" << result.final_freq_hz
              << ", last used f_hz=" << f_used_last << "\n";

    std::cout << "active r=" << result.final_tuning_applied
              << ", active tau_a=" << result.final_tau_applied
              << ", active tau_d=" << result.final_tau_target
              << ", active kb=" << result.final_tuning_target << "\n";

    std::cout << "raw sigma_a=" << result.final_sigma_target
              << ", used sigma_a=" << result.final_sigma_applied
              << ", raw accel_var=" << result.final_accel_variance << "\n";

    std::cout << "disp_scale_m=" << result.wave_params.height
              << " reference Hs, est scale=" << result.final_tuning_applied
              << " (r field shown above)\n";

    std::cout << "===========================================================\n\n";
}

static std::optional<W3dSimulationRunResult>
process_wave_file_for_adaptive_pii_mahony(const std::string& filename,
                                          float dt,
                                          bool with_mag,
                                          bool add_noise,
                                          float mag_odr_hz)
{
    // Reuse the same noise settings as the Kalman simulation example
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
    options.output_suffix_with_mag = "_fusion";
    options.output_suffix_no_mag   = "_fusion_nomag";

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
    }

    return 0;
}
