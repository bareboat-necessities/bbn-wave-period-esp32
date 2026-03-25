extern const float g_std = 9.80665f;
#include "W3dSimCommon.h"

#include <cstdlib>
#include <iostream>
#include <numbers>

ImuNoiseModel make_imu_noise_model(float sigma_white,
                                   float bias_half_range,
                                   float sigma_bias_rw,
                                   unsigned seed)
{
    ImuNoiseModel m;
    m.rng = std::mt19937(seed);
    m.w = std::normal_distribution<float>(0.0f, sigma_white);
    m.n01 = std::normal_distribution<float>(0.0f, 1.0f);
    m.bias0.setZero();
    m.bias_rw.setZero();
    m.sigma_bias_rw = sigma_bias_rw;

    std::uniform_real_distribution<float> ub(-bias_half_range, bias_half_range);
    m.bias0 = Vector3f(ub(m.rng), ub(m.rng), ub(m.rng));
    return m;
}

Vector3f apply_imu_noise(const Vector3f& truth, ImuNoiseModel& m, float dt)
{
    if (m.sigma_bias_rw > 0.0f) {
        const float s = m.sigma_bias_rw * std::sqrt(dt);
        m.bias_rw += Vector3f(s * m.n01(m.rng), s * m.n01(m.rng), s * m.n01(m.rng));
    }
    Vector3f white(m.w(m.rng), m.w(m.rng), m.w(m.rng));
    return truth + (m.bias0 + m.bias_rw) + white;
}

MagNoiseModel make_mag_noise_model(float sigma_white_uT,
                                   float bias_residual_range_uT,
                                   float sigma_bias_rw_uT_sqrt_s,
                                   float scale_err_max,
                                   float cross_axis_max,
                                   float misalign_deg_max,
                                   unsigned seed)
{
    MagNoiseModel m;
    m.rng = std::mt19937(seed);
    m.w_uT = std::normal_distribution<float>(0.0f, sigma_white_uT);
    m.n01 = std::normal_distribution<float>(0.0f, 1.0f);

    std::uniform_real_distribution<float> ub(-bias_residual_range_uT, bias_residual_range_uT);
    m.bias0_uT = Vector3f(ub(m.rng), ub(m.rng), ub(m.rng));
    m.bias_rw_uT.setZero();
    m.sigma_bias_rw_uT_sqrt_s = sigma_bias_rw_uT_sqrt_s;

    std::uniform_real_distribution<float> us(1.0f - scale_err_max, 1.0f + scale_err_max);
    std::uniform_real_distribution<float> uc(-cross_axis_max, cross_axis_max);

    Eigen::Matrix3f A = Eigen::Matrix3f::Identity();
    A(0, 0) = us(m.rng);
    A(1, 1) = us(m.rng);
    A(2, 2) = us(m.rng);

    float a01 = uc(m.rng), a02 = uc(m.rng), a12 = uc(m.rng);
    A(0, 1) = A(1, 0) = a01;
    A(0, 2) = A(2, 0) = a02;
    A(1, 2) = A(2, 1) = a12;

    auto deg2rad = [](float d) { return d * float(std::numbers::pi_v<float> / 180.0); };
    std::uniform_real_distribution<float> ua(-misalign_deg_max, misalign_deg_max);
    float rx = deg2rad(ua(m.rng));
    float ry = deg2rad(ua(m.rng));
    float rz = deg2rad(ua(m.rng));

    auto Rx = [&](float a) {
        Eigen::Matrix3f R;
        float c = std::cos(a), s = std::sin(a);
        R << 1, 0, 0,
             0, c, -s,
             0, s, c;
        return R;
    };
    auto Ry = [&](float a) {
        Eigen::Matrix3f R;
        float c = std::cos(a), s = std::sin(a);
        R << c, 0, s,
             0, 1, 0,
            -s, 0, c;
        return R;
    };
    auto Rz = [&](float a) {
        Eigen::Matrix3f R;
        float c = std::cos(a), s = std::sin(a);
        R << c, -s, 0,
             s, c, 0,
             0, 0, 1;
        return R;
    };

    Eigen::Matrix3f R = Rz(rz) * Ry(ry) * Rx(rx);
    m.Mis = R * A;
    return m;
}

Vector3f apply_mag_noise(const Vector3f& ideal_mag_uT_body, MagNoiseModel& m, float dt_mag)
{
    if (m.sigma_bias_rw_uT_sqrt_s > 0.0f) {
        const float s = m.sigma_bias_rw_uT_sqrt_s * std::sqrt(dt_mag);
        m.bias_rw_uT += Vector3f(s * m.n01(m.rng), s * m.n01(m.rng), s * m.n01(m.rng));
    }
    Vector3f white(m.w_uT(m.rng), m.w_uT(m.rng), m.w_uT(m.rng));
    return (m.Mis * ideal_mag_uT_body) + (m.bias0_uT + m.bias_rw_uT) + white;
}

W3dSimulationRunner::W3dSimulationRunner(W3dSimulationOptions options,
                                         SimulationNoiseModels noise_models,
                                         IW3dFusionAdapter& fusion_adapter)
    : options_(std::move(options)),
      noise_models_(std::move(noise_models)),
      fusion_adapter_(fusion_adapter)
{
}

std::string W3dSimulationRunner::make_output_name(const std::string& filename) const
{
    std::string outname = filename;
    auto pos_prefix = outname.find("wave_data_");
    if (pos_prefix != std::string::npos) {
        outname.replace(pos_prefix, std::string("wave_data_").size(), "w3d_");
    } else {
        outname = "w3d_" + outname;
    }

    auto pos_ext = outname.rfind(".csv");
    const std::string& suffix = options_.with_mag ? options_.output_suffix_with_mag
                                                  : options_.output_suffix_no_mag;
    if (pos_ext != std::string::npos) {
        outname.insert(pos_ext, suffix);
    } else {
        outname += suffix + std::string(".csv");
    }
    return outname;
}

std::optional<W3dSimulationRunResult> W3dSimulationRunner::run(const std::string& filename)
{
    auto parsed = WaveFileNaming::parse_to_params(filename);
    if (!parsed) return std::nullopt;

    auto [kind, type, wp] = *parsed;
    if (kind != FileKind::Data) return std::nullopt;
    if (!(type == WaveType::JONSWAP || type == WaveType::PMSTOKES)) return std::nullopt;

    W3dSimulationRunResult result;
    result.input_name = filename;
    result.output_name = make_output_name(filename);
    result.wave_type = type;
    result.wave_params = wp;

    std::cout << "Processing " << filename << " (type="
              << EnumTraits<WaveType>::to_string(type)
              << ")\n";

    std::ofstream ofs(result.output_name);
    ofs << "time,roll_ref,pitch_ref,yaw_ref,"
        << "disp_ref_x,disp_ref_y,disp_ref_z,"
        << "vel_ref_x,vel_ref_y,vel_ref_z,"
        << "acc_ref_x,acc_ref_y,acc_ref_z,"
        << "roll_est,pitch_est,yaw_est,"
        << "disp_est_x,disp_est_y,disp_est_z,"
        << "vel_est_x,vel_est_y,vel_est_z,"
        << "acc_est_x,acc_est_y,acc_est_z,"
        << "acc_bias_x,acc_bias_y,acc_bias_z,"
        << "gyro_bias_x,gyro_bias_y,gyro_bias_z,"
        << "acc_bias_est_x,acc_bias_est_y,acc_bias_est_z,"
        << "gyro_bias_est_x,gyro_bias_est_y,gyro_bias_est_z,"
        << "mag_bias_x,mag_bias_y,mag_bias_z,"
        << "mag_bias_est_x,mag_bias_est_y,mag_bias_est_z,"
        << "mag_bias_err_x,mag_bias_err_y,mag_bias_err_z,"
        << "tau_applied,sigma_a_applied,R_p0_applied,"
        << "freq_tracker_hz,Tp_tuner_s,accel_var_tuner,"
        << "disp_scale_m,vel_scale_mps,"
        << "dir_phase,"
        << "dir_deg,dir_uncert_deg,dir_conf,dir_amp,"
        << "dir_sign,dir_sign_num,"
        << "dir_vec_x,dir_vec_y,"
        << "dfilt_ax,dfilt_ay\n";

    const Vector3f mag_world_a = MagSim_WMM::mag_world_aero();
    (void)mag_world_a;

    const float mag_dt = 1.0f / options_.mag_odr_hz;
    float mag_phase_s = 0.0f;
    Vector3f mag_body_ned_hold = Vector3f::Zero();

    WaveDataCSVReader reader(filename);
    reader.for_each_record([&](const Wave_Data_Sample& rec) {
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);

        if (options_.add_noise) {
            if (noise_models_.accel_noise) {
                acc_b = apply_imu_noise(acc_b, *noise_models_.accel_noise, options_.dt);
            }
            if (noise_models_.gyro_noise) {
                gyr_b = apply_imu_noise(gyr_b, *noise_models_.gyro_noise, options_.dt);
            }
            for (auto& model : noise_models_.extra_imu_noise_models) {
                model(acc_b, gyr_b, options_.dt);
            }
        }

        Vector3f acc_meas_ned = zu_to_ned(acc_b);
        Vector3f gyr_meas_ned = zu_to_ned(gyr_b);

        float r_ref_out = rec.imu.roll_deg;
        float p_ref_out = rec.imu.pitch_deg;
        float y_ref_out = rec.imu.yaw_deg;

        if (options_.with_mag) {
            mag_phase_s += options_.dt;
            bool mag_tick = false;
            if (mag_phase_s >= mag_dt) {
                while (mag_phase_s >= mag_dt) mag_phase_s -= mag_dt;
                mag_tick = true;
            }
            if (mag_tick) {
                Vector3f mag_b_enu = MagSim_WMM::simulate_mag_from_euler_nautical(r_ref_out, p_ref_out, y_ref_out);
                if (options_.add_noise && noise_models_.mag_noise) {
                    mag_b_enu = apply_mag_noise(mag_b_enu, *noise_models_.mag_noise, mag_dt);
                }
                if (options_.add_noise) {
                    for (auto& model : noise_models_.extra_mag_noise_models) {
                        model(mag_b_enu, mag_dt);
                    }
                }
                mag_body_ned_hold = zu_to_ned(mag_b_enu);
                fusion_adapter_.updateMag(mag_body_ned_hold);
            }
        }

        fusion_adapter_.update(options_.dt, gyr_meas_ned, acc_meas_ned, options_.temperature_c);
        const FilterSnapshot snap = fusion_adapter_.snapshot();

        Vector3f disp_ref(rec.wave.disp_x, rec.wave.disp_y, rec.wave.disp_z);
        Vector3f vel_ref(rec.wave.vel_x, rec.wave.vel_y, rec.wave.vel_z);
        Vector3f acc_ref(rec.wave.acc_x, rec.wave.acc_y, rec.wave.acc_z);

        Vector3f disp_err = snap.disp_est_zu - disp_ref;
        result.errs_x.push_back(disp_err.x());
        result.errs_y.push_back(disp_err.y());
        result.errs_z.push_back(disp_err.z());
        result.ref_x.push_back(disp_ref.x());
        result.ref_y.push_back(disp_ref.y());
        result.ref_z.push_back(disp_ref.z());
        result.errs_roll.push_back(diffDeg(snap.euler_nautical_deg.x(), r_ref_out));
        result.errs_pitch.push_back(diffDeg(snap.euler_nautical_deg.y(), p_ref_out));
        result.errs_yaw.push_back(diffDeg(snap.euler_nautical_deg.z(), y_ref_out));

        const Vector3f acc_bias_true_zu = (options_.add_noise && noise_models_.accel_noise)
            ? (noise_models_.accel_noise->bias0 + noise_models_.accel_noise->bias_rw).eval()
            : Vector3f::Zero().eval();
        const Vector3f gyro_bias_true_zu = (options_.add_noise && noise_models_.gyro_noise)
            ? (noise_models_.gyro_noise->bias0 + noise_models_.gyro_noise->bias_rw).eval()
            : Vector3f::Zero().eval();
        const Vector3f acc_bias_true_ned = zu_to_ned(acc_bias_true_zu);
        const Vector3f gyro_bias_true_ned = zu_to_ned(gyro_bias_true_zu);

        const Vector3f acc_bias_err = snap.acc_bias_est_ned - acc_bias_true_ned;
        const Vector3f gyro_bias_err = snap.gyro_bias_est_ned - gyro_bias_true_ned;

        const Vector3f mag_bias_true_zu = (options_.add_noise && options_.with_mag && noise_models_.mag_noise)
            ? (noise_models_.mag_noise->bias0_uT + noise_models_.mag_noise->bias_rw_uT).eval()
            : Vector3f::Zero().eval();
        const Vector3f mag_bias_true_ned = zu_to_ned(mag_bias_true_zu);
        const Vector3f mag_bias_err = snap.mag_bias_est_ned_uT - mag_bias_true_ned;

        result.accb_err_x.push_back(acc_bias_err.x());
        result.accb_err_y.push_back(acc_bias_err.y());
        result.accb_err_z.push_back(acc_bias_err.z());
        result.gyrb_err_x.push_back(gyro_bias_err.x());
        result.gyrb_err_y.push_back(gyro_bias_err.y());
        result.gyrb_err_z.push_back(gyro_bias_err.z());
        result.magb_err_x.push_back(mag_bias_err.x());
        result.magb_err_y.push_back(mag_bias_err.y());
        result.magb_err_z.push_back(mag_bias_err.z());

        result.accb_true_x.push_back(acc_bias_true_ned.x());
        result.accb_true_y.push_back(acc_bias_true_ned.y());
        result.accb_true_z.push_back(acc_bias_true_ned.z());
        result.gyrb_true_x.push_back(gyro_bias_true_ned.x());
        result.gyrb_true_y.push_back(gyro_bias_true_ned.y());
        result.gyrb_true_z.push_back(gyro_bias_true_ned.z());
        result.magb_true_x.push_back(mag_bias_true_ned.x());
        result.magb_true_y.push_back(mag_bias_true_ned.y());
        result.magb_true_z.push_back(mag_bias_true_ned.z());

        result.freq_hist.push_back(snap.freq_hz);
        result.dir_phase_hist.push_back(snap.direction.phase);
        result.dir_deg_hist.push_back(snap.direction.direction_deg_generator_signed);
        result.dir_unc_hist.push_back(snap.direction.uncertainty_deg);
        result.dir_conf_hist.push_back(snap.direction.confidence);
        result.dir_amp_hist.push_back(snap.direction.amplitude);
        result.dir_sign_num_hist.push_back(snap.direction.sign_num);

        ofs << rec.time << ","
            << r_ref_out << "," << p_ref_out << "," << y_ref_out << ","
            << disp_ref.x() << "," << disp_ref.y() << "," << disp_ref.z() << ","
            << vel_ref.x() << "," << vel_ref.y() << "," << vel_ref.z() << ","
            << acc_ref.x() << "," << acc_ref.y() << "," << acc_ref.z() << ","
            << snap.euler_nautical_deg.x() << "," << snap.euler_nautical_deg.y() << "," << snap.euler_nautical_deg.z() << ","
            << snap.disp_est_zu.x() << "," << snap.disp_est_zu.y() << "," << snap.disp_est_zu.z() << ","
            << snap.vel_est_zu.x() << "," << snap.vel_est_zu.y() << "," << snap.vel_est_zu.z() << ","
            << snap.acc_est_zu.x() << "," << snap.acc_est_zu.y() << "," << snap.acc_est_zu.z() << ","
            << acc_bias_true_ned.x() << "," << acc_bias_true_ned.y() << "," << acc_bias_true_ned.z() << ","
            << gyro_bias_true_ned.x() << "," << gyro_bias_true_ned.y() << "," << gyro_bias_true_ned.z() << ","
            << snap.acc_bias_est_ned.x() << "," << snap.acc_bias_est_ned.y() << "," << snap.acc_bias_est_ned.z() << ","
            << snap.gyro_bias_est_ned.x() << "," << snap.gyro_bias_est_ned.y() << "," << snap.gyro_bias_est_ned.z() << ","
            << mag_bias_true_ned.x() << "," << mag_bias_true_ned.y() << "," << mag_bias_true_ned.z() << ","
            << snap.mag_bias_est_ned_uT.x() << "," << snap.mag_bias_est_ned_uT.y() << "," << snap.mag_bias_est_ned_uT.z() << ","
            << mag_bias_err.x() << "," << mag_bias_err.y() << "," << mag_bias_err.z() << ","
            << snap.tau_applied << ","
            << snap.sigma_applied << ","
            << snap.tuning_applied << ","
            << snap.freq_hz << ","
            << snap.period_sec << ","
            << snap.accel_variance << ","
            << snap.displacement_scale_m << ","
            << snap.velocity_scale_mps << ","
            << snap.direction.phase << "," << snap.direction.direction_deg << "," << snap.direction.uncertainty_deg << ","
            << snap.direction.confidence << "," << snap.direction.amplitude << ","
            << (snap.direction.sign == FORWARD ? "TOWARD" : snap.direction.sign == BACKWARD ? "AWAY" : "UNCERTAIN") << ","
            << snap.direction.sign_num << ","
            << snap.direction.direction_vec.x() << "," << snap.direction.direction_vec.y() << ","
            << snap.direction.filtered_signal.x() << "," << snap.direction.filtered_signal.y() << "\n";

        result.final_tau_target = snap.tau_target;
        result.final_sigma_target = snap.sigma_target;
        result.final_tuning_target = snap.tuning_target;
        result.final_tau_applied = snap.tau_applied;
        result.final_sigma_applied = snap.sigma_applied;
        result.final_tuning_applied = snap.tuning_applied;
        result.final_freq_hz = snap.freq_hz;
        result.final_period_sec = snap.period_sec;
        result.final_accel_variance = snap.accel_variance;
    });

    ofs.close();
    std::cout << "Wrote " << result.output_name << "\n";
    return result;
}

std::vector<std::string> collect_wave_data_files(const std::filesystem::path& directory)
{
    std::vector<std::string> files;
    for (auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().string();
        if (fname.find("wave_data_") == std::string::npos) continue;
        if (auto kind = WaveFileNaming::parse_kind_only(fname); kind && *kind == FileKind::Data) {
            files.push_back(std::move(fname));
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

void print_summary_and_fail_if_needed(const W3dSimulationRunResult& result,
                                      float dt,
                                      const W3dFailureLimits& limits,
                                      const W3dSummaryLabels& labels)
{
    constexpr float RMS_WINDOW_SEC = 60.0f;
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
    const float rad2deg = 180.0f / float(std::numbers::pi_v<float>);
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
              << ", " << labels.target << "=" << result.final_tuning_target << "\n";
    std::cout << "tau_applied=" << result.final_tau_applied
              << ", sigma_applied=" << result.final_sigma_applied
              << ", " << labels.applied << "=" << result.final_tuning_applied << "\n";
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

    const float limit_z = (result.wave_type == WaveType::JONSWAP)
        ? limits.err_limit_percent_z_jonswap
        : limits.err_limit_percent_z_pmstokes;
    const float limit_3d = (result.wave_type == WaveType::JONSWAP)
        ? limits.err_limit_percent_3d_jonswap
        : limits.err_limit_percent_3d_pmstokes;
    auto fail_if = [&](const char* label, float pct, float limit) {
        if (pct > limit) {
            std::cerr << "ERROR: " << label << " RMS above limit (" << pct << "% > " << limit << "%). Failing.\n";
            std::exit(EXIT_FAILURE);
        }
    };

    fail_if("Z", z_pct, limit_z);
    fail_if("3D", pct_3d, limit_3d);

    if (rms_yaw.rms() > limits.err_limit_yaw_deg) {
        std::cerr << "ERROR: Yaw RMS above limit (" << rms_yaw.rms() << " deg > "
                  << limits.err_limit_yaw_deg << " deg). Failing.\n";
        std::exit(EXIT_FAILURE);
    }

    const float accb_z_pct = pct_of_max(accb_rz, acc_true_max_z);
    if (std::isfinite(accb_z_pct) && accb_z_pct > limits.acc_z_bias_percent) {
        std::cerr << "ERROR: accel Z bias error RMS above limit ("
                  << accb_z_pct << "% > " << limits.acc_z_bias_percent
                  << "% of max TRUE Z bias). Failing.\n";
        std::exit(EXIT_FAILURE);
    }
    if (std::isfinite(accb_r3_pct) && accb_r3_pct > limits.bias_3d_percent) {
        std::cerr << "ERROR: 3D accel bias error RMS above limit ("
                  << accb_r3_pct << "% > " << limits.bias_3d_percent
                  << "% of max TRUE bias). Failing.\n";
        std::exit(EXIT_FAILURE);
    }
    if (std::isfinite(gyrb_r3_pct) && gyrb_r3_pct > limits.bias_3d_percent) {
        std::cerr << "ERROR: 3D gyro bias error RMS above limit ("
                  << gyrb_r3_pct << "% > " << limits.bias_3d_percent
                  << "% of max TRUE bias). Failing.\n";
        std::exit(EXIT_FAILURE);
    }
}
