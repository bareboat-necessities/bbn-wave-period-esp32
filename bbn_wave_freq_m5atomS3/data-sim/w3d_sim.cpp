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
#include "WaveFilesSupport.h"    // file reader/parser + naming

using Eigen::Vector3f;
using Eigen::Quaternionf;

// ============================================================
// Coordinate conversions
// Input: simulation data are in nautical Z-up IMU convention
// Filter: aerospace NED convention (Z-down, X forward, Y right)
// Mapping: (x_a, y_a, z_a) = (y_n, x_n, -z_n)
// ============================================================

// Nautical Z-up → Aerospace NED (filter input)
static inline Eigen::Vector3f zu_to_ned(const Eigen::Vector3f& v) {
    return Eigen::Vector3f(v.y(), v.x(), -v.z());
}

// Aerospace NED → Nautical Z-up (for exporting back)
static inline Eigen::Vector3f ned_to_zu(const Eigen::Vector3f& v) {
    return Eigen::Vector3f(v.y(), v.x(), -v.z());
}

// IMU frame → QMEKF frame (expects aerospace convention)
static inline Eigen::Vector3f imu_to_qmekf(const Eigen::Vector3f& v) {
    return zu_to_ned(v);
}

// Quaternion → Euler (deg), ZYX convention with Z-up (nautical frame)
// NOTE: Roll and Pitch signs inverted internally for aerospace
static void quat_to_euler(const Quaternionf &q, float &roll, float &pitch, float &yaw) {
    Eigen::Matrix3f R = q.toRotationMatrix();

    pitch = std::atan2(-R(2,0), std::sqrt(R(0,0)*R(0,0) + R(1,0)*R(1,0)));
    roll  = std::atan2(R(2,1), R(2,2));
    yaw   = std::atan2(R(1,0), R(0,0));

    // Convert to degrees and apply aerospace sign convention
    roll  *= -180.0f / M_PI;   // NEGATED
    pitch *= -180.0f / M_PI;   // NEGATED
    yaw   *=  180.0f / M_PI;   // unchanged
}

// Inverse conversion for Euler angles and orientation errors: aerospace → nautical
static inline void aerospace_to_nautical_euler(float &roll, float &pitch, float &yaw) {
    float old_pitch = pitch, old_roll = roll;
    roll  = -old_roll;   // negate back
    pitch = -old_pitch;  // negate back
    // yaw unchanged
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

    // Errors (exported in nautical Z-up)
    float disp_err_x{}, disp_err_y{}, disp_err_z{};
    float vel_err_x{},  vel_err_y{},  vel_err_z{};
    float acc_err_x{},  acc_err_y{},  acc_err_z{};
    float err_roll{}, err_pitch{}, err_yaw{};
    float angle_err{};
};

void process_wave_file(const std::string &filename, float dt) {
    auto parsed = WaveFileNaming::parse_to_params(filename);
    if (!parsed) return;
    auto [kind, type, wp] = *parsed;

    if (kind != FileKind::Data) return;
    if (!(type == WaveType::JONSWAP || type == WaveType::PMSTOKES)) return;

    std::cout << "Processing " << filename
              << " (" << EnumTraits<WaveType>::to_string(type) << ")\n";

    WaveDataCSVReader reader(filename);

    const Vector3f sigma_a(0.05f,  0.05f,  0.05f);
    const Vector3f sigma_g(0.001f, 0.001f, 0.001f);
    const Vector3f sigma_m(0.10f,  0.10f,  0.10f);
    Kalman3D_Wave<float, true> mekf(sigma_a, sigma_g, sigma_m);

    bool first = true;
    std::vector<OutputRow> rows;

    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);

        Vector3f acc_f = imu_to_qmekf(acc_b);
        Vector3f gyr_f = imu_to_qmekf(gyr_b);

        if (first) {
            mekf.initialize_from_acc(acc_f);
            first = false;
        }

        mekf.time_update(gyr_f, acc_f, dt);
        mekf.measurement_update_acc_only(acc_f);

        auto coeffs = mekf.quaternion().coeffs(); // [x,y,z,w]
        Quaternionf q(coeffs(3), coeffs(0), coeffs(1), coeffs(2));
        float r_est, p_est, y_est;
        quat_to_euler(q, r_est, p_est, y_est);

        // Reference (converted to aerospace convention)
        float r_ref = -rec.imu.pitch_deg;   // NEGATED
        float p_ref = -rec.imu.roll_deg;  // NEGATED
        float y_ref =  rec.imu.yaw_deg;    // unchanged

        // True orientation quaternion from sim
        Quaternionf q_ref =
            Eigen::AngleAxisf(r_ref * M_PI/180.0f, Vector3f::UnitX()) *
            Eigen::AngleAxisf(p_ref * M_PI/180.0f, Vector3f::UnitY()) *
            Eigen::AngleAxisf(y_ref * M_PI/180.0f, Vector3f::UnitZ());

        // Earth magnetic field (NED, normalized)
        static const Vector3f mag_ned_unit(0.39f, 0.0f, -0.92f); // example direction

        // Rotate into body frame
        Vector3f mag_b = q_ref.conjugate() * mag_ned_unit;

        // Add Gaussian noise
        //for (int i = 0; i < 3; i++) {
        //    mag_b(i) += sigma_m(i) * noise(rng);
        //}

        // Convert to filter convention
        Vector3f mag_f = imu_to_qmekf(mag_b);

        // Update filter
        mekf.measurement_update_mag(mag_f);

        // World kinematics (converted to aerospace convention)
        Vector3f disp_ref = zu_to_ned(Vector3f(rec.wave.disp_x, rec.wave.disp_y, rec.wave.disp_z));
        Vector3f vel_ref  = zu_to_ned(Vector3f(rec.wave.vel_x,  rec.wave.vel_y,  rec.wave.vel_z));
        Vector3f acc_ref  = zu_to_ned(Vector3f(rec.wave.acc_x,  rec.wave.acc_y,  rec.wave.acc_z));

        // Estimated world states (aerospace convention)
        Eigen::Vector3f disp_est = mekf.get_position();
        Eigen::Vector3f vel_est  = mekf.get_velocity();
        Eigen::Vector3f acc_est  = mekf.get_world_accel();
        
        // Errors (aerospace convention)
        Vector3f disp_err = disp_est - disp_ref;
        Vector3f vel_err  = vel_est  - vel_ref;
        Vector3f acc_err  = acc_est  - acc_ref;

        float err_roll  = r_est - r_ref;
        float err_pitch = p_est - p_ref;
        float err_yaw   = y_est - y_ref;

        Quaternionf q_ref =
            Eigen::AngleAxisf(r_ref * M_PI/180.0f, Vector3f::UnitX()) *
            Eigen::AngleAxisf(p_ref * M_PI/180.0f, Vector3f::UnitY()) *
            Eigen::AngleAxisf(y_ref * M_PI/180.0f, Vector3f::UnitZ());
        Quaternionf q_est = q.normalized();
        Quaternionf q_err = q_ref.conjugate() * q_est;
        float angle_err = 2.0f * std::acos(std::clamp(q_err.w(), -1.0f, 1.0f)) * 180.0f / M_PI;

        // Convert back to nautical Z-up for export
        Eigen::Vector3f disp_ref_out = ned_to_zu(disp_ref);
        Eigen::Vector3f vel_ref_out  = ned_to_zu(vel_ref);
        Eigen::Vector3f acc_ref_out  = ned_to_zu(acc_ref);
        Eigen::Vector3f disp_est_out = ned_to_zu(disp_est);
        Eigen::Vector3f vel_est_out  = ned_to_zu(vel_est);
        Eigen::Vector3f acc_est_out  = ned_to_zu(acc_est);
        Eigen::Vector3f disp_err_out = ned_to_zu(disp_err);
        Eigen::Vector3f vel_err_out  = ned_to_zu(vel_err);
        Eigen::Vector3f acc_err_out  = ned_to_zu(acc_err);

        float r_ref_out = r_ref;
        float p_ref_out = p_ref;
        float y_ref_out = y_ref;
        float r_est_out = r_est;
        float p_est_out = p_est;
        float y_est_out = y_est;
        aerospace_to_nautical_euler(r_ref_out, p_ref_out, y_ref_out);
        aerospace_to_nautical_euler(r_est_out, p_est_out, y_est_out);

        float err_roll_out  = err_roll;
        float err_pitch_out = err_pitch;
        float err_yaw_out   = err_yaw;
        aerospace_to_nautical_euler(err_roll_out, err_pitch_out, err_yaw_out);

        rows.push_back({
            rec.time,
            r_ref_out, p_ref_out, y_ref_out,
            disp_ref_out.x(), disp_ref_out.y(), disp_ref_out.z(),
            vel_ref_out.x(),  vel_ref_out.y(),  vel_ref_out.z(),
            acc_ref_out.x(),  acc_ref_out.y(),  acc_ref_out.z(),
            rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz,
            rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z,
            r_est_out, p_est_out, y_est_out,
            disp_est_out.x(), disp_est_out.y(), disp_est_out.z(),
            vel_est_out.x(),  vel_est_out.y(),  vel_est_out.z(),
            acc_est_out.x(),  acc_est_out.y(),  acc_est_out.z(),
            disp_err_out.x(), disp_err_out.y(), disp_err_out.z(),
            vel_err_out.x(),  vel_err_out.y(),  vel_err_out.z(),
            acc_err_out.x(),  acc_err_out.y(),  acc_err_out.z(),
            err_roll_out, err_pitch_out, err_yaw_out, angle_err
        });
    });

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
            << r.vel_err_x  << "," << r.vel_err_y << "," << r.vel_err_z << ","
            << r.acc_err_x  << "," << r.acc_err_y << "," << r.acc_err_z << ","
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
