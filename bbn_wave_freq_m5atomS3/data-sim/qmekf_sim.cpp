#include <iostream>
#include <filesystem>
#include <fstream>
#include <Eigen/Dense>
#include "WaveFilesSupport.h"  // file reader/parser + naming
#include "KalmanQMEKF.h"       // Q-MEKF filter

using Eigen::Vector3f;
using Eigen::Quaternionf;

// Quaternion → Euler (deg, XYZ convention)
static void quat_to_euler(const Quaternionf &q, float &roll, float &pitch, float &yaw) {
    float ysqr = q.y() * q.y();

    float t0 = +2.0f * (q.w()*q.x() + q.y()*q.z());
    float t1 = +1.0f - 2.0f * (q.x()*q.x() + ysqr);
    roll = std::atan2(t0, t1) * 180.0f / M_PI;

    float t2 = +2.0f * (q.w()*q.y() - q.z()*q.x());
    t2 = std::clamp(t2, -1.0f, 1.0f);
    pitch = std::asin(t2) * 180.0f / M_PI;

    float t3 = +2.0f * (q.w()*q.z() + q.x()*q.y());
    float t4 = +1.0f - 2.0f * (ysqr + q.z()*q.z());
    yaw = std::atan2(t3, t4) * 180.0f / M_PI;
}

struct OutputRow {
    double t{};
    // Reference Euler
    float roll_ref{}, pitch_ref{}, yaw_ref{};
    // Raw IMU inputs
    float acc_bx{}, acc_by{}, acc_bz{};
    float gyro_x{}, gyro_y{}, gyro_z{};
    // Kalman estimates
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

    // Kalman filter config
    float sigma_a[3] = {0.05f, 0.05f, 0.05f};
    float sigma_g[3] = {0.001f, 0.001f, 0.001f};
    float sigma_m[3] = {0.1f, 0.1f, 0.1f};
    QuaternionMEKF<float, true> mekf(sigma_a, sigma_g, sigma_m);

    bool first = true;
    std::vector<OutputRow> rows;

    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        if (first) {
            // Initialize from first accelerometer vector
            mekf.initialize_from_acc(Vector3f(rec.imu.acc_bx,
                                              rec.imu.acc_by,
                                              rec.imu.acc_bz));
            first = false;
        }

        // Time + measurement updates
        mekf.time_update(Vector3f(rec.imu.gyro_x,
                                  rec.imu.gyro_y,
                                  rec.imu.gyro_z), dt);
        mekf.measurement_update_acc_only(Vector3f(rec.imu.acc_bx,
                                                  rec.imu.acc_by,
                                                  rec.imu.acc_bz));

        // Convert quaternion → Euler
        auto coeffs = mekf.quaternion(); // [x,y,z,w]
        Quaternionf q(coeffs(3), coeffs(0), coeffs(1), coeffs(2));
        float r, p, y;
        quat_to_euler(q, r, p, y);

        rows.push_back({
            rec.time,
            rec.imu.roll_deg, rec.imu.pitch_deg, rec.imu.yaw_deg,
            rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz,
            rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z,
            r, p, y
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
    float dt = 1.0f / 240.0f; // your simulator rate
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
