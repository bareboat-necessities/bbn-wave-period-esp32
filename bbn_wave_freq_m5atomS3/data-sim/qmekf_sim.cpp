#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>   // for std::clamp
#include <string>

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float g_std = 9.80665f; // standard gravity acceleration m/s²

#include "KalmanQMEKF.h"       // Q-MEKF filter
#include "WaveFilesSupport.h"  // file reader/parser + naming
#include "FrameConversions.h"  // conversions + MagSim_WMM

using Eigen::Vector3f;
using Eigen::Quaternionf;

struct OutputRow {
    double t{};
    // Reference Euler (deg, nautical)
    float roll_ref{}, pitch_ref{}, yaw_ref{};
    // Raw IMU inputs (nautical, as read from file)
    float acc_bx{}, acc_by{}, acc_bz{};
    float gyro_x{}, gyro_y{}, gyro_z{};
    // Kalman estimates (deg, nautical)
    float roll_est{}, pitch_est{}, yaw_est{};
};

void process_wave_file(const std::string &filename, float dt, bool with_mag) {
    auto parsed = WaveFileNaming::parse_to_params(filename);
    if (!parsed) return;
    auto [kind, type, wp] = *parsed;

    if (kind != FileKind::Data) return;
    if (!(type == WaveType::JONSWAP || type == WaveType::PMSTOKES)) return;

    std::cout << "Processing " << filename
              << " (" << EnumTraits<WaveType>::to_string(type)
              << ") with_mag=" << (with_mag ? "true" : "false") << "\n";

    WaveDataCSVReader reader(filename);

    const Vector3f sigma_a(0.05f,  0.05f,  0.05f);
    const Vector3f sigma_g(0.001f, 0.001f, 0.001f);
    const Vector3f sigma_m(0.10f,  0.10f,  0.10f);
    QuaternionMEKF<float, true> mekf(sigma_a, sigma_g, sigma_m);

    // World magnetic field in aerospace NED
    const Vector3f mag_world_a = MagSim_WMM::mag_world_aero();

    bool first = true;
    std::vector<OutputRow> rows;

    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);

        // Convert to filter (aerospace NED)
        Vector3f acc_f = zu_to_ned(acc_b);
        Vector3f gyr_f = zu_to_ned(gyr_b);

        // Reference aerospace Euler from simulator
        float r_ref_a = rec.imu.roll_deg;
        float p_ref_a = rec.imu.pitch_deg;
        float y_ref_a = rec.imu.yaw_deg;

        // Simulated magnetometer (if enabled)
        Vector3f mag_f(0,0,0);
        if (with_mag) {
            Vector3f mag_b = MagSim_WMM::simulate_mag_from_euler_aero(r_ref_a, p_ref_a, y_ref_a);
            mag_f = zu_to_ned(mag_b);
        }

        if (first) {
            if (with_mag)
                mekf.initialize_from_acc_mag(acc_f, mag_world_a);
            else
                mekf.initialize_from_acc(acc_f);
            first = false;
        }

        // Propagation
        mekf.time_update(gyr_f, dt);

        // Update: accel always, mag optional
        if (with_mag)
            mekf.measurement_update(acc_f, mag_f);
        else
            mekf.measurement_update_acc_only(acc_f);

        // Filter quaternion → Euler (nautical deg)
        auto coeffs = mekf.quaternion(); // [x,y,z,w]
        Quaternionf q(coeffs(3), coeffs(0), coeffs(1), coeffs(2));
        float r_est, p_est, y_est;
        quat_to_euler_nautical(q, r_est, p_est, y_est);

        // Reference angles (nautical)
        float r_ref = rec.imu.roll_deg;
        float p_ref = rec.imu.pitch_deg;
        float y_ref = rec.imu.yaw_deg;

        rows.push_back({
            rec.time,
            r_ref, p_ref, y_ref,
            rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz,
            rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z,
            r_est, p_est, y_est
        });
    });

    // Output filename
    std::string outname = filename;
    auto pos_prefix = outname.find("wave_data_");
    if (pos_prefix != std::string::npos) {
        outname.replace(pos_prefix, std::string("wave_data_").size(), "qmekf_");
    }
    auto pos_ext = outname.rfind(".csv");
    if (pos_ext != std::string::npos) {
        outname.insert(pos_ext, with_mag ? "_kalman_mag" : "_kalman_nomag");
    } else {
        outname += with_mag ? "_kalman_mag.csv" : "_kalman_nomag.csv";
    }
    
    std::ofstream ofs(outname);
    ofs << "time,"
        << "roll_ref,pitch_ref,yaw_ref,"
        << "acc_bx,acc_by,acc_bz,"
        << "gyro_x,gyro_y,gyro_z,"
        << "roll_est,pitch_est,yaw_est\n";

    for (auto &r : rows) {
        ofs << r.t << ","
            << r.roll_ref << "," << r.pitch_ref << "," << r.yaw_ref << ","
            << r.acc_bx   << "," << r.acc_by    << "," << r.acc_bz << ","
            << r.gyro_x   << "," << r.gyro_y    << "," << r.gyro_z << ","
            << r.roll_est << "," << r.pitch_est << "," << r.yaw_est << "\n";
    }
    ofs.close();

    std::cout << "Wrote " << outname << "\n";
}

int main(int argc, char* argv[]) {
    float dt = 1.0f / 240.0f; // simulator sample rate

    // Default: run with magnetometer
    bool with_mag = true;

    // Command-line toggle: pass "--nomag" to disable
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--nomag") {
            with_mag = false;
        }
    }

    std::cout << "Simulation starting with_mag=" << (with_mag ? "true" : "false") << "\n";

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
