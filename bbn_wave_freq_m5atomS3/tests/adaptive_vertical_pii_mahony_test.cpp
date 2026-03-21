#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "AdaptiveVerticalPIIMahony.h"

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void expect(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

using Filter = marine_obs::AdaptiveVerticalPIIMahony<float, true>;

Filter::Config make_uniform_ocean_config() {
    Filter::Config cfg{};

    cfg.core.observer.r = 0.13f;
    cfg.core.observer.tau_a = 0.95f;
    cfg.core.observer.tau_d = 55.0f;
    cfg.core.observer.kb = 6.0e-5f;
    cfg.core.observer.lambda_b = 8.0e-3f;
    cfg.core.observer.bias_limit = 0.20f;

    cfg.core.adaptation.enabled = true;
    cfg.core.adaptation.min_confidence = 0.22f;
    cfg.core.adaptation.f_disp_ref_hz = 0.12f;
    cfg.core.adaptation.sigma_a_ref = 0.18f;
    cfg.core.adaptation.input_smooth_tau = 4.0f;
    cfg.core.adaptation.param_smooth_tau = 6.0f;
    cfg.core.adaptation.r_freq_exp = 0.80f;
    cfg.core.adaptation.r_sigma_exp = 0.25f;
    cfg.core.adaptation.tau_a_freq_exp = -1.05f;
    cfg.core.adaptation.tau_a_sigma_exp = -0.25f;
    cfg.core.adaptation.tau_d_freq_exp = -0.25f;
    cfg.core.adaptation.tau_d_sigma_exp = -0.15f;
    cfg.core.adaptation.kb_freq_exp = 0.25f;
    cfg.core.adaptation.kb_sigma_exp = 0.50f;
    cfg.core.adaptation.r_min = 0.06f;
    cfg.core.adaptation.r_max = 0.26f;
    cfg.core.adaptation.tau_a_min = 0.35f;
    cfg.core.adaptation.tau_a_max = 2.40f;
    cfg.core.adaptation.tau_d_min = 20.0f;
    cfg.core.adaptation.tau_d_max = 140.0f;
    cfg.core.adaptation.kb_min = 1.0e-5f;
    cfg.core.adaptation.kb_max = 5.0e-4f;

    cfg.core.auto_schedule_from_accel_freq = true;
    cfg.core.auto_schedule_period_s = 0.50f;
    cfg.core.force_enable_adaptation_when_auto_schedule = true;
    cfg.core.fallback_confidence_floor = 0.45f;
    cfg.core.fallback_confidence_when_locked = 0.75f;
    cfg.core.coarse_schedule_blend = 0.65f;
    cfg.core.coarse_schedule_confidence_floor = 0.55f;

    cfg.core.accel_freq_tracker.f_min_hz = 0.045f;
    cfg.core.accel_freq_tracker.f_max_hz = 0.35f;
    cfg.core.accel_freq_tracker.f_init_hz = 0.12f;
    cfg.core.accel_freq_tracker.pre_hp_hz = 0.015f;
    cfg.core.accel_freq_tracker.pre_lp_hz = 0.45f;
    cfg.core.accel_freq_tracker.demod_lp_hz = 0.05f;
    cfg.core.accel_freq_tracker.loop_bandwidth_hz = 0.018f;
    cfg.core.accel_freq_tracker.loop_damping = 1.0f;
    cfg.core.accel_freq_tracker.max_dfdt_hz_per_s = 0.04f;
    cfg.core.accel_freq_tracker.recenter_tau_s = 12.0f;
    cfg.core.accel_freq_tracker.output_smooth_tau_s = 3.0f;
    cfg.core.accel_freq_tracker.power_tau_s = 12.0f;
    cfg.core.accel_freq_tracker.confidence_tau_s = 8.0f;
    cfg.core.accel_freq_tracker.lock_rms_min = 0.01f;
    cfg.core.accel_freq_tracker.enable_coarse_assist = true;
    cfg.core.accel_freq_tracker.coarse_hysteresis_frac = 0.20f;
    cfg.core.accel_freq_tracker.coarse_smooth_tau_s = 4.0f;
    cfg.core.accel_freq_tracker.coarse_pull_tau_s = 2.5f;
    cfg.core.accel_freq_tracker.coarse_timeout_s = 18.0f;

    cfg.mahony_twoKp = 0.45f;
    cfg.mahony_twoKi = 0.015f;
    cfg.gravity_mps2 = 9.80665f;
    cfg.use_mag = false;

    return cfg;
}

Filter::Snapshot run_case(float freq_hz, float accel_amp_mps2, float duration_s) {
    Filter filter(make_uniform_ocean_config());
    constexpr float dt = 0.01f;
    const float omega = 2.0f * static_cast<float>(M_PI) * freq_hz;
    const int steps = static_cast<int>(duration_s / dt);

    for (int i = 0; i < steps; ++i) {
        const float t = i * dt;
        const float vertical_inertial = accel_amp_mps2 * std::sin(omega * t);
        filter.updateIMU(0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 9.80665f + vertical_inertial,
                         dt);
    }

    return filter.snapshot();
}

void test_same_initial_config_learns_different_sea_states() {
    const auto slow = run_case(0.07f, 0.18f, 240.0f);
    const auto fast = run_case(0.22f, 0.85f, 240.0f);

    expect(slow.core.observer.f_disp_filt_hz < 0.11f,
           "slow sea-state case did not learn a low dominant frequency");
    expect(fast.core.observer.f_disp_filt_hz > 0.17f,
           "fast sea-state case did not learn a high dominant frequency");
    expect(fast.core.observer.f_disp_filt_hz > slow.core.observer.f_disp_filt_hz + 0.07f,
           "internal sea-state learning did not separate slow and fast frequency estimates");

    expect(fast.core.observer.r > slow.core.observer.r + 0.03f,
           "observer pole rate did not self-tune upward for the faster sea state");
    expect(slow.core.observer.tau_a > fast.core.observer.tau_a + 0.25f,
           "observer accel smoothing did not self-tune slower for the slower sea state");
    expect(fast.core.observer.sigma_a_filt > slow.core.observer.sigma_a_filt + 0.20f,
           "acceleration sigma estimate did not reflect rougher seas");
}

void test_auto_scheduler_uses_internal_tracker_only() {
    const auto medium = run_case(0.14f, 0.45f, 180.0f);

    expect(medium.core.accel_freq_hz > 0.09f && medium.core.accel_freq_hz < 0.18f,
           "internal frequency tracker did not converge near the driven sea-state frequency");
    expect(medium.core.observer.last_confidence >= 0.45f,
           "auto scheduler never built enough internal confidence to adapt");
    expect(std::abs(medium.core.observer.f_disp_filt_hz - medium.core.accel_freq_hz) < 0.05f,
           "observer scheduling frequency drifted away from the internally learned tracker frequency");
}

}  // namespace

int main() {
    try {
        test_same_initial_config_learns_different_sea_states();
        test_auto_scheduler_uses_internal_tracker_only();
        std::cout << "adaptive_vertical_pii_mahony_test: PASS\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "adaptive_vertical_pii_mahony_test: FAIL: " << ex.what() << '\n';
        return 1;
    }
}
