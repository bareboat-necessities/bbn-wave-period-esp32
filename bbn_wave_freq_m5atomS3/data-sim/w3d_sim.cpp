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

#include "Kalman3D_Wave.h"     // Kalman3D_Wave filter
#include "WaveFilesSupport.h"  // file reader/parser + naming
#include "FrameConversions.h"  // coordinate & quaternion conversions + MagSim_WMM

using Eigen::Vector3f;
using Eigen::Quaternionf;

struct OutputRow {
    double t{};
    float roll_ref{}, pitch_ref{}, yaw_ref{};     // Reference orientation (deg, nautical)
    float disp_ref_x{}, disp_ref_y{}, disp_ref_z{};
    float vel_ref_x{},  vel_ref_y{},  vel_ref_z{};
    float acc_ref_x{},  acc_ref_y{},  acc_ref_z{};
    float acc_bx{}, acc_by{}, acc_bz{};          // Raw IMU inputs (nautical)
    float gyro_x{}, gyro_y{}, gyro_z{};
    float roll_est{}, pitch_est{}, yaw_est{};    // Estimated orientation (deg, nautical)
    float disp_est_x{}, disp_est_y{}, disp_est_z{};
    float vel_est_x{},  vel_est_y{},  vel_est_z{};
    float acc_est_x{},  acc_est_y{},  acc_est_z{};
    float disp_err_x{}, disp_err_y{}, disp_err_z{};
    float vel_err_x{},  vel_err_y{},  vel_err_z{};
    float acc_err_x{},  acc_err_y{},  acc_err_z{};
    float err_roll{}, err_pitch{}, err_yaw{};
    float angle_err{};
};

static void check_init_consistency(const Kalman3D_Wave<float,true>& mekf,
                                   const Eigen::Vector3f& acc_body_meas,
                                   const Eigen::Vector3f& mag_body_meas,
                                   const Eigen::Vector3f& mag_world)
{
    auto qc = mekf.quaternion().coeffs(); // [x,y,z,w]
    Eigen::Quaternionf q(qc(3), qc(0), qc(1), qc(2));
    Eigen::Matrix3f R_bw = q.toRotationMatrix();   // body<-world
    Eigen::Matrix3f R_wb = R_bw.transpose();       // world<-body

    Eigen::Vector3f g_world(0.f, 0.f, -g_std);
    Eigen::Vector3f f_pred_body = R_wb * g_world;
    Eigen::Vector3f m_pred_body = R_wb * mag_world;

    std::cout << "Init Consistency Check\n";
    std::cout << "Accel measured (body) = " << acc_body_meas.transpose()
              << " | predicted (body) = " << f_pred_body.transpose() << "\n";
    std::cout << "Mag   measured (body) = " << mag_body_meas.transpose()
              << " | predicted (body) = " << m_pred_body.transpose() << "\n";
}

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
    Kalman3D_Wave<float, true> mekf(sigma_a, sigma_g, sigma_m);

    // Earth magnetic field in aerospace (NED)
    const Vector3f mag_world_a = MagSim_WMM::mag_world_aero();

    bool first = true;
    std::vector<OutputRow> rows;

    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);

        Vector3f acc_f = zu_to_ned(acc_b);
        Vector3f gyr_f = zu_to_ned(gyr_b);

        // Reference aerospace Euler (from simulator)
        float r_ref_a = rec.imu.roll_deg;
        float p_ref_a = rec.imu.pitch_deg;
        float y_ref_a = rec.imu.yaw_deg;

        // Convert to nautical for output
        float r_ref_out = r_ref_a, p_ref_out = p_ref_a, y_ref_out = y_ref_a;
        aero_to_nautical(r_ref_out, p_ref_out, y_ref_out);

        // Simulated magnetometer [µT]
        Vector3f mag_f(0,0,0);
        if (with_mag) {
            Vector3f mag_b = MagSim_WMM::simulate_mag_from_euler_aero(r_ref_a, p_ref_a, y_ref_a);
            mag_f = zu_to_ned(mag_b);
        }

        if (first) {
            if (with_mag) {
                mekf.initialize_from_acc_mag(acc_f, mag_world_a);
                check_init_consistency(mekf, acc_f, mag_f, mag_world_a);
            } else {
                mekf.initialize_from_acc(acc_f);
            }
            first = false;
        }

        mekf.time_update(gyr_f, acc_f, dt);

        if (with_mag) mekf.measurement_update(acc_f, mag_f);
        else          mekf.measurement_update_acc_only(acc_f);

        auto coeffs = mekf.quaternion().coeffs(); // [x,y,z,w]
        Quaternionf q(coeffs(3), coeffs(0), coeffs(1), coeffs(2));
        float r_est, p_est, y_est;
        quat_to_euler_nautical(q, r_est, p_est, y_est);

        // World kinematics (convert back to nautical for export)
        Vector3f disp_ref = zu_to_ned(Vector3f(rec.wave.disp_x, rec.wave.disp_y, rec.wave.disp_z));
        Vector3f vel_ref  = zu_to_ned(Vector3f(rec.wave.vel_x,  rec.wave.vel_y,  rec.wave.vel_z));
        Vector3f acc_ref  = zu_to_ned(Vector3f(rec.wave.acc_x,  rec.wave.acc_y,  rec.wave.acc_z));

        Vector3f disp_est = mekf.get_position();
        Vector3f vel_est  = mekf.get_velocity();
        Vector3f acc_est  = mekf.get_world_accel();

        Vector3f disp_err = disp_est - disp_ref;
        Vector3f vel_err  = vel_est  - vel_ref;
        Vector3f acc_err  = acc_est  - acc_ref;

        Quaternionf q_ref =
            Eigen::AngleAxisf(r_ref_a * M_PI/180.0f, Vector3f::UnitX()) *
            Eigen::AngleAxisf(p_ref_a * M_PI/180.0f, Vector3f::UnitY()) *
            Eigen::AngleAxisf(y_ref_a * M_PI/180.0f, Vector3f::UnitZ());
        Quaternionf q_err = q_ref.conjugate() * q.normalized();
        float angle_err = 2.0f * std::acos(std::clamp(q_err.w(), -1.0f, 1.0f)) * 180.0f/M_PI;

        rows.push_back({
            rec.time,
            r_ref_out, p_ref_out, y_ref_out,
            ned_to_zu(disp_ref).x(), ned_to_zu(disp_ref).y(), ned_to_zu(disp_ref).z(),
            ned_to_zu(vel_ref).x(),  ned_to_zu(vel_ref).y(),  ned_to_zu(vel_ref).z(),
            ned_to_zu(acc_ref).x(),  ned_to_zu(acc_ref).y(),  ned_to_zu(acc_ref).z(),
            rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz,
            rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z,
            r_est, p_est, y_est,
            ned_to_zu(disp_est).x(), ned_to_zu(disp_est).y(), ned_to_zu(disp_est).z(),
            ned_to_zu(vel_est).x(),  ned_to_zu(vel_est).y(),  ned_to_zu(vel_est).z(),
            ned_to_zu(acc_est).x(),  ned_to_zu(acc_est).y(),  ned_to_zu(acc_est).z(),
            ned_to_zu(disp_err).x(), ned_to_zu(disp_err).y(), ned_to_zu(disp_err).z(),
            ned_to_zu(vel_err).x(),  ned_to_zu(vel_err).y(),  ned_to_zu(vel_err).z(),
            ned_to_zu(acc_err).x(),  ned_to_zu(acc_err).y(),  ned_to_zu(acc_err).z(),
            r_est - r_ref_out, p_est - p_ref_out, y_est - y_ref_out,
            angle_err
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

int main(int argc, char* argv[]) {
    float dt = 1.0f / 240.0f;

    bool with_mag = true;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--nomag") with_mag = false;
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
