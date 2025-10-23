#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <random>
#include <cmath>

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ZERO_CROSSINGS_SCALE          1.0f
#define ZERO_CROSSINGS_DEBOUNCE_TIME  0.12f
#define ZERO_CROSSINGS_STEEPNESS_TIME 0.21f

#define FREQ_GUESS 0.3f   // frequency guess

const float g_std = 9.80665f;     // standard gravity acceleration m/s²

const float FAIL_ERR_LIMIT_PERCENT_X_HIGH = 40.0f;
const float FAIL_ERR_LIMIT_PERCENT_Y_HIGH = 40.0f;
const float FAIL_ERR_LIMIT_PERCENT_Z_HIGH = 13.0f;

const float FAIL_ERR_LIMIT_PERCENT_X_LOW  = 40.0f;
const float FAIL_ERR_LIMIT_PERCENT_Y_LOW  = 40.0f;
const float FAIL_ERR_LIMIT_PERCENT_Z_LOW  = 13.0f;

constexpr float RMS_WINDOW_SEC = 60.0f;  // RMS window

//  Project headers
#include "WaveFilesSupport.h"
#include "FrameConversions.h"
#include "SeaStateFusionFilter.h"

using Eigen::Vector3f;
using Eigen::Quaternionf;

//  RMS helper
class RMSReport {
public:
    inline void add(float value) { sum_sq_ += value * value; count_++; }
    inline float rms() const { return count_ ? std::sqrt(sum_sq_ / float(count_)) : NAN; }
private:
    float sum_sq_ = 0.0f;
    size_t count_ = 0;
};

//  Noise model
bool add_noise = true;
struct NoiseModel {
    std::default_random_engine rng;
    std::normal_distribution<float> dist;
    Vector3f bias;
};
NoiseModel make_noise_model(float sigma, float bias_range, unsigned seed) {
    NoiseModel m{std::default_random_engine(seed),
                 std::normal_distribution<float>(0.0f, sigma),
                 Vector3f::Zero()};
    std::uniform_real_distribution<float> ub(-bias_range, bias_range);
    m.bias = Vector3f(ub(m.rng), ub(m.rng), ub(m.rng));
    return m;
}
Vector3f apply_noise(const Vector3f& v, NoiseModel& m) {
    return v - m.bias + Vector3f(m.dist(m.rng), m.dist(m.rng), m.dist(m.rng));
}

//  Example wave parameter list
const std::vector<WaveParameters> waveParamsList = {
    {3.0f,   0.27f, static_cast<float>(M_PI/3.0), 25.0f},
    {5.7f,   1.5f,  static_cast<float>(M_PI/1.5), 25.0f},
    {8.5f,   4.0f,  static_cast<float>(M_PI/6.0), 25.0f},
    {11.4f,  8.5f,  static_cast<float>(M_PI/2.5), 25.0f}
};
int wave_index_from_height(float height) {
    for (size_t i = 0; i < waveParamsList.size(); i++)
        if (std::abs(waveParamsList[i].height - height) < 1e-3f) return int(i);
    return -1;
}

//  Main processing
static void process_wave_file_for_tracker(const std::string &filename,
                                          float dt,
                                          bool with_mag)
{
    auto parsed = WaveFileNaming::parse_to_params(filename);
    if (!parsed) return;
    auto [kind, type, wp] = *parsed;
    if (kind != FileKind::Data) return;
    if (!(type == WaveType::JONSWAP || type == WaveType::PMSTOKES)) return;

    std::string outname = filename;
    auto pos_prefix = outname.find("wave_data_");
    if (pos_prefix != std::string::npos)
        outname.replace(pos_prefix, std::string("wave_data_").size(), "w3d_");
    else outname = "w3d_" + outname;
    auto pos_ext = outname.rfind(".csv");
    if (pos_ext != std::string::npos)
        outname.insert(pos_ext, with_mag ? "_fusion" : "_fusion_nomag");
    else outname += (with_mag ? "_fusion" : "_fusion_nomag") + std::string(".csv");

    std::cout << "Processing " << filename << " (type="
              << EnumTraits<WaveType>::to_string(type) << ")\n";

    std::ofstream ofs(outname);
    ofs << "time,roll_ref,pitch_ref,yaw_ref,"
        << "disp_ref_x,disp_ref_y,disp_ref_z,"
        << "vel_ref_x,vel_ref_y,vel_ref_z,"
        << "acc_ref_x,acc_ref_y,acc_ref_z,"
        << "roll_est,pitch_est,yaw_est,"
        << "disp_est_x,disp_est_y,disp_est_z,"
        << "vel_est_x,vel_est_y,vel_est_z,"
        << "acc_est_x,acc_est_y,acc_est_z,"
        << "tau_applied,sigma_a_applied,R_S_applied,"
        << "freq_tracker_hz,Tp_tuner_s,accel_var_tuner\n";

    // Initialize unified fusion filter
    using Fusion = SeaStateFusionFilter<TrackerType::KALMANF>;
    Fusion filter(with_mag);

    const Vector3f sigma_a_init(0.30f, 0.30f, 0.30f);
    const Vector3f sigma_g(0.00134f, 0.00134f, 0.00134f);
    const Vector3f sigma_m(0.3f, 0.3f, 0.3f);
    filter.initialize(sigma_a_init, sigma_g, sigma_m);

    const Vector3f mag_world_a = MagSim_WMM::mag_world_aero();

    static NoiseModel accel_noise = make_noise_model(0.03f, 0.02f, 1234);
    static NoiseModel gyro_noise  = make_noise_model(0.001f, 0.0004f, 5678);

    bool first = true;
    WaveDataCSVReader reader(filename);

    std::vector<float> errs_x, errs_y, errs_z, errs_roll, errs_pitch, errs_yaw;

    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        // Body-frame noisy sensors
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);
        if (add_noise) {
            acc_b = apply_noise(acc_b, accel_noise);
            gyr_b = apply_noise(gyr_b, gyro_noise);
        }

// Transform IMU readings (specific force) from Z-up to NED
Vector3f acc_meas_ned = zu_to_ned(acc_b);  // includes +g, as a real accelerometer would
Vector3f gyr_meas_ned = zu_to_ned(gyr_b);

// Feed the filter with specific force (includes gravity)
filter.updateTime(dt, gyr_meas_ned, acc_meas_ned);
        
        // Reference Euler (nautical ENU/Z-up)
        float r_ref_out = rec.imu.roll_deg;
        float p_ref_out = rec.imu.pitch_deg;
        float y_ref_out = rec.imu.yaw_deg;

        // Simulated magnetometer
        Vector3f mag_f(0,0,0);
        if (with_mag) {
            Vector3f mag_b_enu = MagSim_WMM::simulate_mag_from_euler_nautical(
                r_ref_out, p_ref_out, y_ref_out);
            mag_f = zu_to_ned(mag_b_enu);
        }

        if (first) {
            filter.initialize_from_acc(acc_f);
            first = false;
        }

        // Unified fusion update
        filter.updateTime(dt, gyr_f, acc_f);
        if (with_mag && rec.time >= MAG_DELAY_SEC)
            filter.updateMag(mag_f);

        // Extract results
        Vector3f disp_ref(rec.wave.disp_x, rec.wave.disp_y, rec.wave.disp_z);
        Vector3f vel_ref (rec.wave.vel_x,  rec.wave.vel_y,  rec.wave.vel_z);
        Vector3f acc_ref (rec.wave.acc_x,  rec.wave.acc_y,  rec.wave.acc_z);

        Vector3f disp_est = ned_to_zu(filter.mekf().get_position());
        Vector3f vel_est  = ned_to_zu(filter.mekf().get_velocity());
        // Remove gravity from world acceleration before comparing with true motion
        Vector3f acc_est  = ned_to_zu(filter.mekf().get_world_accel() - Vector3f(0, 0, g_std));
        
        Eigen::Vector3f eul_est = filter.getEulerNautical(); // roll,pitch,yaw (deg)

        Vector3f disp_err = disp_est - disp_ref;
        Vector3f vel_err  = vel_est  - vel_ref;
        Vector3f acc_err  = acc_est  - acc_ref;

        errs_x.push_back(disp_err.x());
        errs_y.push_back(disp_err.y());
        errs_z.push_back(disp_err.z());
        errs_roll.push_back(eul_est.x() - r_ref_out);
        errs_pitch.push_back(eul_est.y() - p_ref_out);
        errs_yaw.push_back(eul_est.z() - y_ref_out);

        // Write CSV
        ofs << rec.time << ","
            << r_ref_out << "," << p_ref_out << "," << y_ref_out << ","
            << disp_ref.x() << "," << disp_ref.y() << "," << disp_ref.z() << ","
            << vel_ref.x()  << "," << vel_ref.y()  << "," << vel_ref.z()  << ","
            << acc_ref.x()  << "," << acc_ref.y()  << "," << acc_ref.z()  << ","
            << eul_est.x() << "," << eul_est.y() << "," << eul_est.z() << ","
            << disp_est.x() << "," << disp_est.y() << "," << disp_est.z() << ","
            << vel_est.x()  << "," << vel_est.y()  << "," << vel_est.z() << ","
            << acc_est.x()  << "," << acc_est.y()  << "," << acc_est.z() << ","
            << filter.getTauApplied() << ","
            << filter.getSigmaApplied() << ","
            << filter.getRSApplied() << ","
            << filter.getFreqHz() << ","
            << filter.getPeriodSec() << ","
            << filter.getAccelVariance() << "\n";
    });

    ofs.close();
    std::cout << "Wrote " << outname << "\n";

    //  RMS summary (last 60 s)
    int N_last = static_cast<int>(RMS_WINDOW_SEC / dt);
    if (errs_z.size() > static_cast<size_t>(N_last)) {
        size_t start = errs_z.size() - N_last;

        RMSReport rms_x, rms_y, rms_z, rms_roll, rms_pitch, rms_yaw;
        for (size_t i = start; i < errs_z.size(); ++i) {
            rms_x.add(errs_x[i]);
            rms_y.add(errs_y[i]);
            rms_z.add(errs_z[i]);
            rms_roll.add(errs_roll[i]);
            rms_pitch.add(errs_pitch[i]);
            rms_yaw.add(errs_yaw[i]);
        }

        float x_rms = rms_x.rms(), y_rms = rms_y.rms(), z_rms = rms_z.rms();
        float x_pct = 100.f * x_rms / wp.height;
        float y_pct = 100.f * y_rms / wp.height;
        float z_pct = 100.f * z_rms / wp.height;

        std::cout << "=== Last 60 s RMS summary for " << outname << " ===\n";
        std::cout << "XYZ RMS (m): X=" << x_rms << " Y=" << y_rms << " Z=" << z_rms << "\n";
        std::cout << "XYZ RMS (%Hs): X=" << x_pct << "% Y=" << y_pct << "% Z=" << z_pct << "% (Hs=" << wp.height << ")\n";
        std::cout << "Angles RMS (deg): Roll=" << rms_roll.rms()
                  << " Pitch=" << rms_pitch.rms()
                  << " Yaw=" << rms_yaw.rms() << "\n";
        std::cout << "=============================================\n\n";
    }
}

//  Main
int main(int argc, char* argv[]) {
    float dt = 1.0f / 240.0f;
    bool with_mag = true;
    add_noise = true;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--nomag") with_mag = false;
        else if (arg == "--no-noise") add_noise = false;
    }

    std::cout << "Simulation starting with_mag=" << (with_mag ? "true" : "false")
              << ", mag_delay=" << MAG_DELAY_SEC
              << " sec, noise=" << (add_noise ? "true" : "false") << "\n";

    std::vector<std::string> files;
    for (auto &entry : std::filesystem::directory_iterator(".")) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().string();
        if (fname.find("wave_data_") == std::string::npos) continue;
        if (auto kind = WaveFileNaming::parse_kind_only(fname);
            kind && *kind == FileKind::Data) {
            files.push_back(std::move(fname));
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto& fname : files)
        process_wave_file_for_tracker(fname, dt, with_mag);

    return 0;
}
