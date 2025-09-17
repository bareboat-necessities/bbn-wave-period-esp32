#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>   // for std::clamp
#include <string>
#include <vector>
#include <random>

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float g_std = 9.80665f;     // standard gravity acceleration m/s²
const float MAG_DELAY_SEC = 5.0f; // delay before enabling magnetometer

#include "Kalman3D_Wave.h"     // Kalman3D_Wave filter
#include "WaveFilesSupport.h"  // file reader/parser + naming
#include "FrameConversions.h"  // coordinate & quaternion conversions + MagSim_WMM

using Eigen::Vector3f;
using Eigen::Quaternionf;

// Noise model (accel & gyro noisy by default; disable with --no-noise)
bool add_noise = true;

struct NoiseModel {
    std::default_random_engine rng;
    std::normal_distribution<float> dist;
    Vector3f bias;
};

NoiseModel make_noise_model(float sigma, float bias_range, unsigned seed) {
    NoiseModel m{std::default_random_engine(seed), std::normal_distribution<float>(0.0f, sigma), Vector3f::Zero()};
    std::uniform_real_distribution<float> ub(-bias_range, bias_range);
    m.bias = Vector3f(ub(m.rng), ub(m.rng), ub(m.rng));
    return m;
}

Vector3f apply_noise(const Vector3f& v, NoiseModel& m) {
    return v - m.bias + Vector3f(m.dist(m.rng), m.dist(m.rng), m.dist(m.rng));
}

struct OutputRow {
    double t{};
    float roll_ref{}, pitch_ref{}, yaw_ref{};
    float disp_ref_x{}, disp_ref_y{}, disp_ref_z{};
    float vel_ref_x{},  vel_ref_y{},  vel_ref_z{};
    float acc_ref_x{},  acc_ref_y{},  acc_ref_z{};
    float acc_bx{}, acc_by{}, acc_bz{};
    float gyro_x{}, gyro_y{}, gyro_z{};
    float roll_est{}, pitch_est{}, yaw_est{};
    float disp_est_x{}, disp_est_y{}, disp_est_z{};
    float vel_est_x{},  vel_est_y{},  vel_est_z{};
    float acc_est_x{},  acc_est_y{},  acc_est_z{};
    float disp_err_x{}, disp_err_y{}, disp_err_z{};
    float vel_err_x{},  vel_err_y{},  vel_err_z{};
    float acc_err_x{},  acc_err_y{},  acc_err_z{};
    float err_roll{}, err_pitch{}, err_yaw{};
    float angle_err{};
    // Noisy IMU
    float acc_noisy_x{}, acc_noisy_y{}, acc_noisy_z{};
    float gyro_noisy_x{}, gyro_noisy_y{}, gyro_noisy_z{};
    // True injected biases
    float acc_bias_x{}, acc_bias_y{}, acc_bias_z{};
    float gyro_bias_x{}, gyro_bias_y{}, gyro_bias_z{};
    // Estimated biases
    float acc_bias_est_x{}, acc_bias_est_y{}, acc_bias_est_z{};
    float gyro_bias_est_x{}, gyro_bias_est_y{}, gyro_bias_est_z{};
};

void process_wave_file(const std::string &filename, float dt, bool with_mag) {
    auto parsed = WaveFileNaming::parse_to_params(filename);
    if (!parsed) return;
    auto [kind, type, wp] = *parsed;
    if (kind != FileKind::Data) return;
    if (!(type == WaveType::JONSWAP || type == WaveType::PMSTOKES)) return;

    std::cout << "Processing " << filename
              << " (" << EnumTraits<WaveType>::to_string(type)
              << ") with_mag=" << (with_mag ? "true" : "false")
              << ", noise=" << (add_noise ? "true" : "false") << "\n";

    WaveDataCSVReader reader(filename);

    const Vector3f sigma_a(0.04f,  0.04f,  0.04f);
    const Vector3f sigma_g(0.00134f, 0.00134f, 0.00134f);
    const Vector3f sigma_m(0.30f,  0.30f,  0.30f);
    Kalman3D_Wave<float, true, true> mekf(sigma_a, sigma_g, sigma_m);

    const Vector3f mag_world_a = MagSim_WMM::mag_world_aero();

    // Noise models (fixed params, seeded differently)
    static NoiseModel accel_noise = make_noise_model(0.04f, 0.05f, 1234);
    static NoiseModel gyro_noise  = make_noise_model(0.001f, 0.0004f, 5678);

    bool first = true;
    bool mag_enabled = false;
    std::vector<OutputRow> rows;

    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);

        // Apply optional noise/bias
        Vector3f acc_noisy = acc_b;
        Vector3f gyr_noisy = gyr_b;
        if (add_noise) {
            acc_noisy = apply_noise(acc_b, accel_noise);
            gyr_noisy = apply_noise(gyr_b, gyro_noise);
        }

        Vector3f acc_f = zu_to_ned(acc_noisy);
        Vector3f gyr_f = zu_to_ned(gyr_noisy);

        // Reference Euler (nautical ENU/Z-up)
        float r_ref_out = rec.imu.roll_deg;
        float p_ref_out = rec.imu.pitch_deg;
        float y_ref_out = rec.imu.yaw_deg;

        // Simulated magnetometer
        Vector3f mag_f(0,0,0);
        if (with_mag) {
            Vector3f mag_b_enu =
                MagSim_WMM::simulate_mag_from_euler_nautical(r_ref_out, p_ref_out, y_ref_out);
            mag_f = zu_to_ned(mag_b_enu);
        }

        if (first) {
            mekf.initialize_from_acc(acc_f);
            first = false;
        }

        mekf.time_update(gyr_f, dt);

        if (with_mag && rec.time >= MAG_DELAY_SEC) {
            if (!mag_enabled) {
                mekf.set_mag_world_ref(mag_world_a);
                mekf.measurement_update_mag_only(mag_f);
                mag_enabled = true;
            }
            mekf.measurement_update_acc_only(acc_f);
            mekf.measurement_update_mag_only(mag_f);
        } else {
            mekf.measurement_update_acc_only(acc_f);
        }

        auto coeffs = mekf.quaternion().coeffs();
        Quaternionf q(coeffs(3), coeffs(0), coeffs(1), coeffs(2));

        float r_est_a, p_est_a, y_est_a;
        quat_to_euler_aero(q, r_est_a, p_est_a, y_est_a);
        float r_est = r_est_a, p_est = p_est_a, y_est = y_est_a;
        aero_to_nautical(r_est, p_est, y_est);

        Vector3f disp_ref(rec.wave.disp_x, rec.wave.disp_y, rec.wave.disp_z);
        Vector3f vel_ref (rec.wave.vel_x,  rec.wave.vel_y,  rec.wave.vel_z);
        Vector3f acc_ref (rec.wave.acc_x,  rec.wave.acc_y,  rec.wave.acc_z);

        Vector3f disp_est = ned_to_zu(mekf.get_position());
        Vector3f vel_est  = ned_to_zu(mekf.get_velocity());
        Vector3f acc_est  = ned_to_zu(mekf.get_world_accel());

        Vector3f disp_err = disp_est - disp_ref;
        Vector3f vel_err  = vel_est  - vel_ref;
        Vector3f acc_err  = acc_est  - acc_ref;

        Quaternionf q_est_naut =
            Eigen::AngleAxisf(r_est*M_PI/180.0f, Vector3f::UnitX()) *
            Eigen::AngleAxisf(p_est*M_PI/180.0f, Vector3f::UnitY()) *
            Eigen::AngleAxisf(y_est*M_PI/180.0f, Vector3f::UnitZ());

        Quaternionf q_ref =
            Eigen::AngleAxisf(r_ref_out*M_PI/180.0f, Vector3f::UnitX()) *
            Eigen::AngleAxisf(p_ref_out*M_PI/180.0f, Vector3f::UnitY()) *
            Eigen::AngleAxisf(y_ref_out*M_PI/180.0f, Vector3f::UnitZ());

        Quaternionf q_err = q_ref.conjugate() * q_est_naut.normalized();
        float angle_err = 2.0f * std::acos(std::clamp(q_err.w(), -1.0f, 1.0f)) * 180.0f/M_PI;

        // Estimated biases
        Vector3f bacc_est = mekf.get_acc_bias();
        Vector3f bgyr_est = mekf.gyroscope_bias();

        rows.push_back({
            rec.time,
            r_ref_out, p_ref_out, y_ref_out,
            disp_ref.x(), disp_ref.y(), disp_ref.z(),
            vel_ref.x(),  vel_ref.y(),  vel_ref.z(),
            acc_ref.x(),  acc_ref.y(),  acc_ref.z(),
            rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz,
            rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z,
            r_est, p_est, y_est,
            disp_est.x(), disp_est.y(), disp_est.z(),
            vel_est.x(),  vel_est.y(),  vel_est.z(),
            acc_est.x(),  acc_est.y(),  acc_est.z(),
            disp_err.x(), disp_err.y(), disp_err.z(),
            vel_err.x(),  vel_err.y(),  vel_err.z(),
            acc_err.x(),  acc_err.y(),  acc_err.z(),
            r_est - r_ref_out, p_est - p_ref_out, y_est - y_ref_out,
            angle_err,
            acc_noisy.x(), acc_noisy.y(), acc_noisy.z(),
            gyr_noisy.x(), gyr_noisy.y(), gyr_noisy.z(),
            accel_noise.bias.x(), accel_noise.bias.y(), accel_noise.bias.z(),
            gyro_noise.bias.x(),  gyro_noise.bias.y(),  gyro_noise.bias.z(),
            bacc_est.x(), bacc_est.y(), bacc_est.z(),
            bgyr_est.x(), bgyr_est.y(), bgyr_est.z()
        });
    });

    std::string outname = filename;
    auto pos_prefix = outname.find("wave_data_");
    if (pos_prefix != std::string::npos) {
        outname.replace(pos_prefix, std::string("wave_data_").size(), "w3d_");
    }
    auto pos_ext = outname.rfind(".csv");
    if (pos_ext != std::string::npos) {
        outname.insert(pos_ext, with_mag ? "_w3d" : "_w3d_nomag");
    } else {
        outname += with_mag ? "_w3d.csv" : "_w3d_nomag.csv";
    }

    std::ofstream ofs(outname);
    ofs << "time,"
        << "roll_ref,pitch_ref,yaw_ref,"
        << "disp_ref_x,disp_ref_y,disp_ref_z,"
        << "vel_ref_x,vel_ref_y,vel_ref_z,"
        << "acc_ref_x,acc_ref_y,acc_ref_z,"
        << "acc_bx,acc_by,acc_bz,"
        << "gyro_x,gyro_y,gyro_z,"
        << "roll_est,pitch_est,yaw_est,"
        << "disp_est_x,disp_est_y,disp_est_z,"
        << "vel_est_x,vel_est_y,vel_est_z,"
        << "acc_est_x,acc_est_y,acc_est_z,"
        << "disp_err_x,disp_err_y,disp_err_z,"
        << "vel_err_x,vel_err_y,vel_err_z,"
        << "acc_err_x,acc_err_y,acc_err_z,"
        << "err_roll,err_pitch,err_yaw,angle_err,"
        << "acc_noisy_x,acc_noisy_y,acc_noisy_z,"
        << "gyro_noisy_x,gyro_noisy_y,gyro_noisy_z,"
        << "acc_bias_x,acc_bias_y,acc_bias_z,"
        << "gyro_bias_x,gyro_bias_y,gyro_bias_z,"
        << "acc_bias_est_x,acc_bias_est_y,acc_bias_est_z,"
        << "gyro_bias_est_x,gyro_bias_est_y,gyro_bias_est_z\n";

    for (auto &r : rows) {
        ofs << r.t << ","
            << r.roll_ref << "," << r.pitch_ref << "," << r.yaw_ref << ","
            << r.disp_ref_x << "," << r.disp_ref_y << "," << r.disp_ref_z << ","
            << r.vel_ref_x  << "," << r.vel_ref_y  << "," << r.vel_ref_z  << ","
            << r.acc_ref_x  << "," << r.acc_ref_y  << "," << r.acc_ref_z  << ","
            << r.acc_bx << "," << r.acc_by << "," << r.acc_bz << ","
            << r.gyro_x << "," << r.gyro_y << "," << r.gyro_z << ","
            << r.roll_est << "," << r.pitch_est << "," << r.yaw_est << ","
            << r.disp_est_x << "," << r.disp_est_y << "," << r.disp_est_z << ","
            << r.vel_est_x  << "," << r.vel_est_y  << "," << r.vel_est_z << ","
            << r.acc_est_x  << "," << r.acc_est_y  << "," << r.acc_est_z << ","
            << r.disp_err_x << "," << r.disp_err_y << "," << r.disp_err_z << ","
            << r.vel_err_x  << "," << r.vel_err_y << "," << r.vel_err_z << ","
            << r.acc_err_x  << "," << r.acc_err_y << "," << r.acc_err_z << ","
            << r.err_roll   << "," << r.err_pitch  << "," << r.err_yaw << ","
            << r.angle_err << ","
            << r.acc_noisy_x << "," << r.acc_noisy_y << "," << r.acc_noisy_z << ","
            << r.gyro_noisy_x << "," << r.gyro_noisy_y << "," << r.gyro_noisy_z << ","
            << r.acc_bias_x << "," << r.acc_bias_y << "," << r.acc_bias_z << ","
            << r.gyro_bias_x << "," << r.gyro_bias_y << "," << r.gyro_bias_z << ","
            << r.acc_bias_est_x << "," << r.acc_bias_est_y << "," << r.acc_bias_est_z << ","
            << r.gyro_bias_est_x << "," << r.gyro_bias_est_y << "," << r.gyro_bias_est_z
            << "\n";
    }
    ofs.close();

    std::cout << "Wrote " << outname << "\n";
}

int main(int argc, char* argv[]) {
    float dt = 1.0f / 240.0f;

    bool with_mag = true;
    add_noise = true;  // default: noisy

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--nomag") with_mag = false;
        if (arg == "--no-noise") add_noise = false;
    }

    std::cout << "Simulation starting with_mag=" << (with_mag ? "true" : "false")
              << ", mag_delay=" << MAG_DELAY_SEC << " sec"
              << ", noise=" << (add_noise ? "true" : "false") << "\n";

    for (auto &entry : std::filesystem::directory_iterator(".")) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().string();
        if (fname.find("wave_data_") == std::string::npos) continue;
        auto kind = WaveFileNaming::parse_kind_only(fname);
        if (kind && *kind == FileKind::Data) {
            process_wave_file(fname, dt, with_mag);
        }
    }
    return 0;
}
