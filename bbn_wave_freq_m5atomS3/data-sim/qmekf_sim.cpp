#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>   // for std::clamp

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float g_std = 9.80665f; // standard gravity acceleration m/s²

#include "KalmanQMEKF.h"       // Q-MEKF filter
#include "WaveFilesSupport.h"  // file reader/parser + naming
#include "FrameConversions.h"  // coordinate & quaternion conversions

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

void process_wave_file(const std::string &filename, float dt) {
    // Parse metadata
    auto parsed = WaveFileNaming::parse_to_params(filename);
    if (!parsed) return;
    auto [kind, type, wp] = *parsed;

    // Restrict to JONSWAP + PMSTOKES
    if (kind != FileKind::Data) return;
    if (!(type == WaveType::JONSWAP || type == WaveType::PMSTOKES)) return;

    std::cout << "Processing " << filename
              << " (" << EnumTraits<WaveType>::to_string(type) << ")\n";

    WaveDataCSVReader reader(filename);

    // Kalman filter config (std devs — squared internally)
    const Vector3f sigma_a(0.05f,  0.05f,  0.05f);
    const Vector3f sigma_g(0.001f, 0.001f, 0.001f);
    const Vector3f sigma_m(0.10f,  0.10f,  0.10f);
    QuaternionMEKF<float, true> mekf(sigma_a, sigma_g, sigma_m);

    bool first = true;
    std::vector<OutputRow> rows;

    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        // Build Eigen vectors from CSV (nautical IMU/body frame)
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);

        // Convert to QMEKF filter frame (aerospace NED)
        Vector3f acc_f = zu_to_ned(acc_b);
        Vector3f gyr_f = zu_to_ned(gyr_b);

        if (first) {
            mekf.initialize_from_acc(acc_f);
            first = false;
        }

        // Time + measurement updates
        mekf.time_update(gyr_f, dt);
        mekf.measurement_update_acc_only(acc_f);

        // Filter quaternion → Euler (nautical deg)
        auto coeffs = mekf.quaternion(); // [x,y,z,w]
        Quaternionf q(coeffs(3), coeffs(0), coeffs(1), coeffs(2));
        float r_est, p_est, y_est;
        quat_to_euler_nautical(q, r_est, p_est, y_est);

        // Reference angles from simulator (already nautical, deg)
        float r_ref = rec.imu.roll_deg;
        float p_ref = rec.imu.pitch_deg;
        float y_ref = rec.imu.yaw_deg;

        rows.push_back({
            rec.time,
            r_ref, p_ref, y_ref,                  // reference (nautical)
            rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz, // raw IMU (nautical)
            rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z, // raw gyro (nautical)
            r_est, p_est, y_est                   // estimated (converted to nautical)
        });
    });

    // Output filename: replace prefix "wave_data_" with "qmekf_", then add "_kalman"
    std::string outname = filename;

    auto pos_prefix = outname.find("wave_data_");
    if (pos_prefix != std::string::npos) {
        outname.replace(pos_prefix, std::string("wave_data_").size(), "qmekf_");
    }

    auto pos_ext = outname.rfind(".csv");
    if (pos_ext != std::string::npos) {
        outname.insert(pos_ext, "_kalman");
    } else {
        outname += "_kalman.csv";
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

int main() {
    float dt = 1.0f / 240.0f; // simulator sample rate
    for (auto &entry : std::filesystem::directory_iterator(".")) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().string();
        if (fname.find("wave_data_") == std::string::npos) continue;
        auto kind = WaveFileNaming::parse_kind_only(fname);
        if (kind && *kind == FileKind::Data) {
            process_wave_file(fname, dt);
        }
    }
    return 0;
}
