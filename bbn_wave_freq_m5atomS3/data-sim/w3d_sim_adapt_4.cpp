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
#include "SeaStateFusionFilter_OU_II.h"

using Eigen::Vector3f;

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
    using Fusion = SeaStateFusion_OU_II<TrackerType::KALMANF>;
    mutable Fusion fusion_;
    Fusion::Config cfg_{};
};


static constexpr W3dFailureLimits FAIL_LIMITS{
    .err_limit_percent_z_jonswap = 9.97f,
    .err_limit_percent_z_pmstokes = 9.9f,
    .err_limit_yaw_deg = 3.98f,
    .err_limit_percent_3d_jonswap = 50.0f,
    .err_limit_percent_3d_pmstokes = 50.0f,
    .acc_z_bias_percent = 28.0f,
    .bias_3d_percent = 250.0f,
};

static constexpr W3dSummaryLabels SUMMARY_LABELS{
    .target = "p0_S_target",
    .applied = "p0_S_applied",
};

static void process_wave_file_for_tracker(const std::string& filename, float dt, bool with_mag)
{
    constexpr float MAG_ODR_HZ = 25.0f;
    auto result = process_wave_file_for_tracker<FusionAdapter4>(filename, dt, with_mag, add_noise, MAG_ODR_HZ, "_fusion_ou2", "_fusion_ou2_nomag");
    if (!result) return;
    print_summary_and_fail_if_needed(*result, dt, FAIL_LIMITS, SUMMARY_LABELS);
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

    const auto files = collect_wave_data_files(".");

    for (const auto& fname : files) {
        process_wave_file_for_tracker(fname, dt, with_mag);
    }
    return 0;
}

