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

using Eigen::Vector3f;
using Eigen::Quaternionf;

// Quaternion → Euler (deg), ZYX convention with Z-up (nautical frame)
static void quat_to_euler(const Quaternionf &q, float &roll, float &pitch, float &yaw) {
    // Rotation matrix from quaternion
    Eigen::Matrix3f R = q.toRotationMatrix();

    // Match simulator’s extraction (same as getEulerAngles)
    pitch = std::atan2(-R(2,0), std::sqrt(R(0,0)*R(0,0) + R(1,0)*R(1,0)));
    roll  = std::atan2(R(2,1), R(2,2));
    yaw   = std::atan2(R(1,0), R(0,0));

    // Convert to degrees
    roll  *= 180.0f / M_PI;
    pitch *= 180.0f / M_PI;
    yaw   *= 180.0f / M_PI;
}

// IMU frame → QMEKF frame
// Fix: only flip Z (up→down) to match the filter's gravity convention.
static inline Vector3f imu_to_qmekf(const Vector3f& v) {
    return Vector3f(v.x(), v.y(), -v.z());
}

struct OutputRow {
    double t{};
    // Reference Euler (deg)
    float roll_ref{}, pitch_ref{}, yaw_ref{};
    // Raw IMU inputs
    float acc_bx{}, acc_by{}, acc_bz{};
    float gyro_x{}, gyro_y{}, gyro_z{};
    // Kalman estimates (deg)
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
        // Build Eigen vectors from CSV (IMU/body frame)
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);

        // Convert to QMEKF filter frame (flip Z only)
        Vector3f acc_f = imu_to_qmekf(acc_b);
        Vector3f gyr_f = imu_to_qmekf(gyr_b);

        if (first) {
            mekf.initialize_from_acc(acc_f);
            first = false;
        }

        // Time + measurement updates
        mekf.time_update(gyr_f, dt);
        mekf.measurement_update_acc_only(acc_f);

        // Filter quaternion → Euler (deg)
        auto coeffs = mekf.quaternion(); // [x,y,z,w]
        Quaternionf q(coeffs(3), coeffs(0), coeffs(1), coeffs(2));
        float r_est, p_est, y_est;
        quat_to_euler(q, r_est, p_est, y_est);

        // Reference angles from simulator (already in deg)
        float r_ref = rec.imu.roll_deg;
        float p_ref = rec.imu.pitch_deg;
        float y_ref = rec.imu.yaw_deg;

        rows.push_back({
            rec.time,
            r_ref, p_ref, y_ref,                  // reference
            rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz,
            rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z,
            r_est, p_est, y_est                   // estimated
        });
    });

    // Output filename: preserve original name + add "_kalman" before ".csv"
    std::string outname = filename;
    auto pos = outname.rfind(".csv");
    if (pos != std::string::npos) {
        outname.insert(pos, "_kalman");
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
        auto kind = WaveFileNaming::parse_kind_only(fname);
        if (kind && *kind == FileKind::Data) {
            process_wave_file(fname, dt);
        }
    }
    return 0;
}

