#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>   // for std::clamp

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float g_std = 9.80665f; // standard gravity acceleration m/s²

#include "Kalman3D_Wave.h"       // Kalman3D_Wave filter
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
// Only flip Z (up→down) to match the filter's gravity convention.
static inline Vector3f imu_to_qmekf(const Vector3f& v) {
    return Vector3f(v.x(), v.y(), -v.z());
}

struct OutputRow {
    double t{};

    // Reference orientation (deg)
    float roll_ref{}, pitch_ref{}, yaw_ref{};

    // Reference world kinematics
    float disp_ref_x{}, disp_ref_y{}, disp_ref_z{};
    float vel_ref_x{},  vel_ref_y{},  vel_ref_z{};
    float acc_ref_x{},  acc_ref_y{},  acc_ref_z{};

    // Raw IMU inputs
    float acc_bx{}, acc_by{}, acc_bz{};
    float gyro_x{}, gyro_y{}, gyro_z{};

    // Kalman orientation estimates (deg)
    float roll_est{}, pitch_est{}, yaw_est{};

    // Estimated world kinematics
    float disp_est_x{}, disp_est_y{}, disp_est_z{};
    float vel_est_x{},  vel_est_y{},  vel_est_z{};
    float acc_est_x{},  acc_est_y{},  acc_est_z{};

    // Errors
    float disp_err_x{}, disp_err_y{}, disp_err_z{};
    float vel_err_x{},  vel_err_y{},  vel_err_z{};
    float acc_err_x{},  acc_err_y{},  acc_err_z{};
    float err_roll{}, err_pitch{}, err_yaw{};
    float angle_err{};
};

void process_wave_file(const std::string &filename, float dt) {
    // Parse metadata
    auto parsed = WaveFileNaming::parse_to_params(filename);
    if (!parsed) return;
    auto [kind, type, wp] = *parsed;

    if (kind != FileKind::Data) return;
    if (!(type == WaveType::JONSWAP || type == WaveType::PMSTOKES)) return;

    std::cout << "Processing " << filename
              << " (" << EnumTraits<WaveType>::to_string(type) << ")\n";

    WaveDataCSVReader reader(filename);

    // Kalman filter config
    const Vector3f sigma_a(0.05f,  0.05f,  0.05f);
    const Vector3f sigma_g(0.001f, 0.001f, 0.001f);
    const Vector3f sigma_m(0.10f,  0.10f,  0.10f);
    Kalman3D_Wave<float, true> mekf(sigma_a, sigma_g, sigma_m);

    bool first = true;
    std::vector<OutputRow> rows;

    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        // Build Eigen vectors from CSV (IMU/body frame)
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);

        // Convert to QMEKF frame (flip Z only)
        Vector3f acc_f = imu_to_qmekf(acc_b);
        Vector3f gyr_f = imu_to_qmekf(gyr_b);

        if (first) {
            mekf.initialize_from_acc(acc_f);
            first = false;
        }

        // Time + measurement updates
        mekf.time_update(gyr_f, acc_f, dt);
        mekf.measurement_update_acc_only(acc_f);

        // Quaternion → Euler (deg)
        auto coeffs = mekf.quaternion().coeffs(); // [x,y,z,w]
        Quaternionf q(coeffs(3), coeffs(0), coeffs(1), coeffs(2));
        float r_est, p_est, y_est;
        quat_to_euler(q, r_est, p_est, y_est);

        // Reference (from generator, already in deg + world kinematics)
        float r_ref = rec.imu.roll_deg;
        float p_ref = rec.imu.pitch_deg;
        float y_ref = rec.imu.yaw_deg;
        Vector3f disp_ref(rec.wave.disp_x, rec.wave.disp_y, rec.wave.disp_z);
        Vector3f vel_ref (rec.wave.vel_x,  rec.wave.vel_y,  rec.wave.vel_z);
        Vector3f acc_ref (rec.wave.acc_x,  rec.wave.acc_y,  rec.wave.acc_z);

        // Estimated world states
        Vector3f disp_est = mekf.get_position();
        Vector3f vel_est  = mekf.get_velocity();
        Vector3f acc_est  = mekf.get_world_accel();

        // Errors
        Vector3f disp_err = disp_est - disp_ref;
        Vector3f vel_err  = vel_est  - vel_ref;
        Vector3f acc_err  = acc_est  - acc_ref;

        float err_roll  = r_est - r_ref;
        float err_pitch = p_est - p_ref;
        float err_yaw   = y_est - y_ref;

        // Quaternion angular error
        Quaternionf q_ref =
            Eigen::AngleAxisf(r_ref * M_PI/180.0f, Vector3f::UnitX()) *
            Eigen::AngleAxisf(p_ref * M_PI/180.0f, Vector3f::UnitY()) *
            Eigen::AngleAxisf(y_ref * M_PI/180.0f, Vector3f::UnitZ());
        Quaternionf q_est = q.normalized();
        Quaternionf q_err = q_ref.conjugate() * q_est;
        float angle_err = 2.0f * std::acos(std::clamp(q_err.w(), -1.0f, 1.0f)) * 180.0f / M_PI;

        rows.push_back({
            rec.time,
            r_ref, p_ref, y_ref,
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
            err_roll, err_pitch, err_yaw, angle_err
        });
    });

    // Output filename
    std::string outname = filename;
    auto pos_prefix = outname.find("wave_data_");
    if (pos_prefix != std::string::npos) {
        outname.replace(pos_prefix, std::string("wave_data_").size(), "w3d_");
    }
    auto pos_ext = outname.rfind(".csv");
    if (pos_ext != std::string::npos) {
        outname.insert(pos_ext, "_w3d");
    } else {
        outname += "_w3d.csv";
    }    

    // Write CSV
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
        << "err_roll,err_pitch,err_yaw,angle_err\n";

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
            << r.vel_est_x  << "," << r.vel_est_y  << "," << r.vel_est_z  << ","
            << r.acc_est_x  << "," << r.acc_est_y  << "," << r.acc_est_z  << ","
            << r.disp_err_x << "," << r.disp_err_y << "," << r.disp_err_z << ","
            << r.vel_err_x  << "," << r.vel_err_y  << "," << r.vel_err_z  << ","
            << r.acc_err_x  << "," << r.acc_err_y  << "," << r.acc_err_z  << ","
            << r.err_roll   << "," << r.err_pitch  << "," << r.err_yaw << ","
            << r.angle_err << "\n";
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
