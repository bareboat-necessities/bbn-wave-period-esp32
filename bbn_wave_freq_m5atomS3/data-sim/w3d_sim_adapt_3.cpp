#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <random>
#include <cmath>

/*
    Copyright (c) 2025  Mikhail Grushinskiy  
*/

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
const float FAIL_ERR_LIMIT_PERCENT_Z_HIGH = 22.0f;

const float FAIL_ERR_LIMIT_PERCENT_X_LOW  = 40.0f;
const float FAIL_ERR_LIMIT_PERCENT_Y_LOW  = 40.0f;
const float FAIL_ERR_LIMIT_PERCENT_Z_LOW  = 22.0f;

const float FAIL_ERR_LIMIT_YAW_DEG = 10.0f;  

constexpr float RMS_WINDOW_SEC = 60.0f;  // RMS window

//  Project headers
#include "WaveFilesSupport.h"
#include "FrameConversions.h"
#include "SeaStateFusionFilter.h"

using Eigen::Vector3f;
using Eigen::Quaternionf;

inline float wrapDeg(float a){
    a = std::fmod(a + 180.0f, 360.0f);
    if (a < 0) a += 360.0f;
    return a - 180.0f;
}
inline float diffDeg(float est_deg, float ref_deg){
    return wrapDeg(est_deg - ref_deg);
}

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

struct ImuNoiseModel {
    std::mt19937 rng;
    std::normal_distribution<float> w;      // white noise (per-sample std)
    std::normal_distribution<float> n01;    // N(0,1) for RW
    Vector3f bias0;                         // fixed bias
    Vector3f bias_rw;                       // drifting component
    float sigma_bias_rw = 0.0f;             // units / sqrt(s)
};

ImuNoiseModel make_imu_noise_model(float sigma_white,
                                  float bias_half_range,
                                  float sigma_bias_rw,
                                  unsigned seed)
{
    ImuNoiseModel m;
    m.rng  = std::mt19937(seed);
    m.w    = std::normal_distribution<float>(0.0f, sigma_white);
    m.n01  = std::normal_distribution<float>(0.0f, 1.0f);
    m.bias0.setZero();
    m.bias_rw.setZero();
    m.sigma_bias_rw = sigma_bias_rw;

    std::uniform_real_distribution<float> ub(-bias_half_range, bias_half_range);
    m.bias0 = Vector3f(ub(m.rng), ub(m.rng), ub(m.rng));
    return m;
}

// measurement = truth + bias + noise
Vector3f apply_imu_noise(const Vector3f& truth, ImuNoiseModel& m, float dt)
{
    if (m.sigma_bias_rw > 0.0f) {
        const float s = m.sigma_bias_rw * std::sqrt(dt);
        m.bias_rw += Vector3f(s * m.n01(m.rng), s * m.n01(m.rng), s * m.n01(m.rng));
    }
    Vector3f white(m.w(m.rng), m.w(m.rng), m.w(m.rng));
    return truth + (m.bias0 + m.bias_rw) + white;
}

struct MagNoiseModel {
    std::mt19937 rng;
    std::normal_distribution<float> dist;
    Vector3f bias;          // hard-iron bias [uT]
    Eigen::Matrix3f Mis;    // soft-iron + misalignment matrix
};

MagNoiseModel make_mag_noise_model(float sigma_uT,
                                   float bias_range_uT,
                                   float scale_err_max,
                                   float misalign_deg_max,
                                   unsigned seed)
{
    MagNoiseModel m;
    m.rng  = std::mt19937(seed);
    m.dist = std::normal_distribution<float>(0.0f, sigma_uT);

    // Hard-iron bias
    std::uniform_real_distribution<float> ub(-bias_range_uT, bias_range_uT);
    m.bias = Vector3f(ub(m.rng), ub(m.rng), ub(m.rng));

    // Soft-iron + misalignment: build as R * S
    m.Mis.setIdentity();

    // Scale errors (diagonal)
    std::uniform_real_distribution<float> us(1.0f - scale_err_max,
                                             1.0f + scale_err_max);
    Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
    S(0,0) = us(m.rng);
    S(1,1) = us(m.rng);
    S(2,2) = us(m.rng);

    // Small random rotations around each axis
    auto deg2rad = [](float d){ return d * float(M_PI/180.0); };
    std::uniform_real_distribution<float> ua(-misalign_deg_max, misalign_deg_max);
    float rx = deg2rad(ua(m.rng));
    float ry = deg2rad(ua(m.rng));
    float rz = deg2rad(ua(m.rng));

    auto Rx = [&](float a) {
        Eigen::Matrix3f R;
        float c = std::cos(a), s = std::sin(a);
        R << 1, 0, 0,
             0, c,-s,
             0, s, c;
        return R;
    };
    auto Ry = [&](float a) {
        Eigen::Matrix3f R;
        float c = std::cos(a), s = std::sin(a);
        R <<  c, 0, s,
              0, 1, 0,
             -s, 0, c;
        return R;
    };
    auto Rz = [&](float a) {
        Eigen::Matrix3f R;
        float c = std::cos(a), s = std::sin(a);
        R << c,-s, 0,
             s, c, 0,
             0, 0, 1;
        return R;
    };

    Eigen::Matrix3f R = Rz(rz) * Ry(ry) * Rx(rx);
    m.Mis = R * S;  // apply scale then misalignment

    return m;
}

Vector3f apply_mag_noise(const Vector3f& ideal_mag_uT_body, MagNoiseModel& m)
{
    // Hard/soft iron
    Vector3f distorted = m.Mis * ideal_mag_uT_body + m.bias;

    // Add white noise
    Vector3f n(m.dist(m.rng), m.dist(m.rng), m.dist(m.rng));
    return distorted + n;
}

//  Example wave parameter list
const std::vector<WaveParameters> waveParamsList = {
    {3.0f,   0.27f, static_cast<float>(M_PI/3.0), 30.0f},
    {5.7f,   1.5f,  static_cast<float>(M_PI/1.5), 30.0f},
    {8.5f,   4.0f,  static_cast<float>(M_PI/6.0), 30.0f},
    {11.4f,  8.5f,  static_cast<float>(M_PI/2.5), 30.0f}
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
    if (pos_ext != std::string::npos) {
        outname.insert(pos_ext, with_mag ? "_fusion" : "fusion_nomag");
    } else {
        outname += (with_mag ? "_fusion" : "_fusion_nomag") + std::string(".csv");
    }

    std::cout << "Processing " << filename << " (type="
              << EnumTraits<WaveType>::to_string(type)
              << ")\n";

    std::ofstream ofs(outname);
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
        << "tau_applied,sigma_a_applied,R_S_applied,"
        << "freq_tracker_hz,Tp_tuner_s,accel_var_tuner\n";

    // Initialize unified fusion filter
    using Fusion = SeaStateFusionFilter<TrackerType::KALMANF>;
    Fusion filter(with_mag);

    // Magnetic reference (same each run)
    const Vector3f mag_world_a = MagSim_WMM::mag_world_aero();

// --- BMI270-like noise (per-axis), “normal mode” order ---
// White noise (per-sample RMS, per axis)
const float acc_sigma = 1.51e-3f * g_std;                 // 1.51 mg-rms -> ~0.0148 m/s^2   [oai_citation:6‡Bosch Sensortec](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi270-ds000.pdf)
const float gyr_sigma = 0.09f * float(M_PI/180.0f);        // 0.09 dps-rms -> ~0.00157 rad/s  [oai_citation:7‡Bosch Sensortec](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi270-ds000.pdf)

// “Decently calibrated” residual constant bias half-ranges
const float acc_bias_range = 5e-3f * g_std;                // 5 mg -> ~0.049 m/s^2
const float gyr_bias_range = 0.05f * float(M_PI/180.0f);    // 0.05 dps -> ~0.00087 rad/s

// Optional very-slow drift (keep tiny; set 0 if you don’t want drift)
const float acc_bias_rw = 0.0005f;   // m/s^2 / sqrt(s)
const float gyr_bias_rw = 0.00001f;  // rad/s / sqrt(s)

ImuNoiseModel accel_noise = make_imu_noise_model(acc_sigma, acc_bias_range, acc_bias_rw, 1234);
ImuNoiseModel gyro_noise  = make_imu_noise_model(gyr_sigma, gyr_bias_range, gyr_bias_rw, 5678);

    // Magnetometer noise model (units: uT)
    MagNoiseModel mag_noise = make_mag_noise_model(
        0.3f,   // sigma uT per sample
        15.0f,  // hard-iron bias uT
        0.0f,  // scale error up 
        0.0f,   // misalignment deg
        9012    // seed
    );

    // Filter
    const Vector3f sigma_a_init(2.1*acc_sigma, 2.1*acc_sigma, 2.1*acc_sigma);
    const Vector3f sigma_g(2.1*gyr_sigma, 2.1*gyr_sigma, 2.1*gyr_sigma);
    const Vector3f sigma_m(0.15f, 0.15f, 0.15f);
    filter.initialize(sigma_a_init, sigma_g, sigma_m);
    
    bool first = true;
    bool mag_ref_set = false;  
    WaveDataCSVReader reader(filename);
    
    std::vector<float> errs_x, errs_y, errs_z, errs_roll, errs_pitch, errs_yaw;

    int sample_idx = -1;
    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        ++sample_idx;

        // Body-frame raw sensors (Z-up body from CSV)
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);

        const bool use_noise = add_noise;
        if (use_noise) {
            acc_b = apply_imu_noise(acc_b, accel_noise, dt);
            gyr_b = apply_imu_noise(gyr_b, gyro_noise, dt);       
        }

        // Map body Z-up -> body NED axes (still BODY, not world)
        Vector3f acc_meas_ned = zu_to_ned(acc_b);  // specific force (includes +g)
        Vector3f gyr_meas_ned = zu_to_ned(gyr_b);  // body angular rate

        // Reference Euler (nautical, ENU/Z-up)
        float r_ref_out = rec.imu.roll_deg;
        float p_ref_out = rec.imu.pitch_deg;
        float y_ref_out = rec.imu.yaw_deg;

        // Simulated magnetometer (BODY, then axis-map to NED body)
        Vector3f mag_body_ned(0,0,0);
        if (with_mag) {
            // Ideal body-frame mag (Z-up ENU body)
            Vector3f mag_b_enu = MagSim_WMM::simulate_mag_from_euler_nautical(
                r_ref_out, p_ref_out, y_ref_out);

            // Add realistic noise / bias / misalignment in ENU body frame
            if (add_noise) {
                mag_b_enu = apply_mag_noise(mag_b_enu, mag_noise);
            }

            // Axis map Z-up ENU body -> NED body
            mag_body_ned = zu_to_ned(mag_b_enu);
        }
        
        // First-step init
        if (first) {
            // Attitude from accel
            filter.initialize_from_acc(acc_meas_ned);  
            first = false;
        }

        // One-time world magnetic reference before using magnetometer
        if (with_mag && !mag_ref_set && rec.time >= MAG_DELAY_SEC) {
            filter.mekf().set_mag_world_ref(mag_world_a);
            mag_ref_set = true; 
        }

        // One time update per sample (propagate + accel update)
        filter.updateTime(dt, gyr_meas_ned, acc_meas_ned, 35.0f);

        // Yaw correction after mag is available
        if (with_mag && rec.time >= MAG_DELAY_SEC && (sample_idx % 3 == 0))
            filter.updateMag(mag_body_ned);
        
        // Reference (world Z-up)
        Vector3f disp_ref(rec.wave.disp_x, rec.wave.disp_y, rec.wave.disp_z);
        Vector3f vel_ref (rec.wave.vel_x, rec.wave.vel_y, rec.wave.vel_z);
        Vector3f acc_ref (rec.wave.acc_x, rec.wave.acc_y, rec.wave.acc_z);

        // Estimates: MEKF state is world NED -> convert to Z-up for CSV
        Vector3f disp_est = ned_to_zu(filter.mekf().get_position());
        Vector3f vel_est  = ned_to_zu(filter.mekf().get_velocity());
        Vector3f acc_est  = ned_to_zu(filter.mekf().get_world_accel());

        Eigen::Vector3f eul_est = filter.getEulerNautical(); // roll,pitch,yaw (deg)

        Vector3f disp_err = disp_est - disp_ref;
        Vector3f vel_err  = vel_est  - vel_ref;
        Vector3f acc_err  = acc_est  - acc_ref;

        errs_x.push_back(disp_err.x());
        errs_y.push_back(disp_err.y());
        errs_z.push_back(disp_err.z());
        errs_roll.push_back(diffDeg(eul_est.x(), r_ref_out));
        errs_pitch.push_back(diffDeg(eul_est.y(), p_ref_out));
        errs_yaw.push_back(diffDeg(eul_est.z(), y_ref_out));
        
        Vector3f acc_bias_true  = accel_noise.bias0 + accel_noise.bias_rw;
        Vector3f gyro_bias_true = gyro_noise.bias0 + gyro_noise.bias_rw;        
        Vector3f acc_bias_est   = filter.mekf().get_acc_bias();
        Vector3f gyro_bias_est  = filter.mekf().gyroscope_bias();
    
        // CSV row
        ofs << rec.time << ","
            << r_ref_out << "," << p_ref_out << "," << y_ref_out << ","
            << disp_ref.x() << "," << disp_ref.y() << "," << disp_ref.z() << ","
            << vel_ref.x()  << "," << vel_ref.y()  << "," << vel_ref.z()  << ","
            << acc_ref.x()  << "," << acc_ref.y()  << "," << acc_ref.z()  << ","
            << eul_est.x() << "," << eul_est.y() << "," << eul_est.z() << ","
            << disp_est.x() << "," << disp_est.y() << "," << disp_est.z() << ","
            << vel_est.x()  << "," << vel_est.y()  << "," << vel_est.z() << ","
            << acc_est.x()  << "," << acc_est.y()  << "," << acc_est.z() << ","
            << acc_bias_true.x() << "," << acc_bias_true.y() << "," << acc_bias_true.z() << ","
            << gyro_bias_true.x() << "," << gyro_bias_true.y() << "," << gyro_bias_true.z() << ","
            << acc_bias_est.x()  << "," << acc_bias_est.y()  << "," << acc_bias_est.z()  << ","
            << gyro_bias_est.x() << "," << gyro_bias_est.y() << "," << gyro_bias_est.z() << ","
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

        // Extended diagnostic summary
        float tau_target   = filter.getTauTarget();
        float sigma_target = filter.getSigmaTarget();
        float RS_target    = filter.getRSTarget();

        float tau_applied   = filter.getTauApplied();
        float sigma_applied = filter.getSigmaApplied();
        float RS_applied    = filter.getRSApplied();

        float f_hz          = filter.getFreqHz();
        float Tp_tuner      = filter.getPeriodSec();
        float accel_var     = filter.getAccelVariance();

        std::cout << "tau_target=" << tau_target
                  << ", sigma_target=" << sigma_target
                  << ", RS_target=" << RS_target << "\n";
        std::cout << "tau_applied=" << tau_applied
                  << ", sigma_applied=" << sigma_applied
                  << ", RS_applied=" << RS_applied << "\n";
        std::cout << "f_hz=" << f_hz
                  << ", Tp_tuner=" << Tp_tuner
                  << ", accel_var=" << accel_var << "\n";
        std::cout << "=============================================\n\n";

        // Failure criteria 
        float limit_x = (type == WaveType::JONSWAP) ? FAIL_ERR_LIMIT_PERCENT_X_HIGH : FAIL_ERR_LIMIT_PERCENT_X_LOW;
        float limit_y = (type == WaveType::JONSWAP) ? FAIL_ERR_LIMIT_PERCENT_Y_HIGH : FAIL_ERR_LIMIT_PERCENT_Y_LOW;
        float limit_z = (type == WaveType::JONSWAP) ? FAIL_ERR_LIMIT_PERCENT_Z_HIGH : FAIL_ERR_LIMIT_PERCENT_Z_LOW;

        auto fail_if = [&](const char* axis, float pct, float limit) {
            if (pct > limit) {
                std::cerr << "ERROR: " << axis << " RMS above limit ("
                          << pct << "% > " << limit << "%). Failing.\n";
                std::exit(EXIT_FAILURE);
            }
        };
        fail_if("X", x_pct, limit_x);
        fail_if("Y", y_pct, limit_y);
        fail_if("Z", z_pct, limit_z);

        if (rms_yaw.rms() > FAIL_ERR_LIMIT_YAW_DEG) {
            std::cerr << "ERROR: Yaw RMS above limit ("
                      << rms_yaw.rms() << " deg > " << FAIL_ERR_LIMIT_YAW_DEG
                      << " deg). Failing.\n";
            std::exit(EXIT_FAILURE);
        }
    }
}

//  Main
int main(int argc, char* argv[]) {
    float dt = 1.0f / 240.0f;
    bool with_mag = true;
    add_noise = true;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--nomag") {
            with_mag = false;
        } else if (arg == "--no-noise") {
            add_noise = false;
        }
    }

    std::cout << "Simulation starting with_mag=" << (with_mag ? "true" : "false")
              << ", mag_delay=" << MAG_DELAY_SEC
              << " sec, noise=" << (add_noise ? "true" : "false")
              << "\n";

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
