#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>   // std::clamp
#include <string>
#include <vector>
#include <map>
#include <random>
#include <deque>
#include <cmath>

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float g_std = 9.80665f;     // standard gravity acceleration m/s²
const float MAG_DELAY_SEC = 5.0f; // delay before enabling magnetometer

const float FAIL_ERR_LIMIT_PERCENT_HIGH = 1000.0f;
const float FAIL_ERR_LIMIT_PERCENT_LOW  = 1000.0f;

// Global variable set from command line
float R_S_base_global = 1.1f;   // default

// Rolling stats window [s] for RMS and online variance
constexpr float RMS_WINDOW_SEC = 60.0f;

// Online estimation warmup before applying to MEKF [s]
constexpr float ONLINE_TUNE_WARMUP_SEC = 15.0f;

// Adaptation rate (fixed, per second). Effective per-step alpha is 1-exp(-RATE*dt)
constexpr float ADAPT_RATE_PER_SEC = 0.1f;  

// Stability clamps
constexpr float MIN_SIGMA_A = 0.1f;    // m/s^2
constexpr float MAX_SIGMA_A = 5.0f;    // m/s^2
constexpr float MIN_FREQ_HZ = 0.1f;    // Hz
constexpr float MAX_FREQ_HZ = 8.0f;    // Hz
constexpr float MIN_TAU_S   = 0.5f;
constexpr float MAX_TAU_S   = 5.0f;

// Trackers smoothing is handled by your helpers (FrequencySmoother + Kalman smoother) in estimate_freq()

#include "Kalman3D_Wave.h"
#include "WaveFilesSupport.h"
#include "FrameConversions.h"

#include "AranovskiyFilter.h"
#include "KalmANF.h"
#include "FrequencySmoother.h"
#include "KalmanForWaveBasic.h"
#include "KalmanWaveNumStableAlt.h"
#include "SchmittTriggerFrequencyDetector.h"
#include "KalmanSmoother.h"
#include "KalmanWaveDirection.h"
#include "WaveFilters.h"
#include "SeaStateRegularity.h"

using Eigen::Vector3f;
using Eigen::Quaternionf;

// ---------------------------
// Helpers
// ---------------------------

// Inline RMS accumulator (float)
class RMSReport {
public:
    inline void add(float value) { sum_sq_ += value * value; count_++; }
    inline float rms() const { return count_ ? std::sqrt(sum_sq_ / float(count_)) : NAN; }
private:
    float sum_sq_ = 0.0f;
    size_t count_ = 0;
};

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

// Example test cases (height → index mapping)
const std::vector<WaveParameters> waveParamsList = {
    {3.0f,   0.27f, static_cast<float>(M_PI/3.0), 25.0f},
    {5.7f,   1.5f,  static_cast<float>(M_PI/1.5), 25.0f},
    {8.5f,   4.0f,  static_cast<float>(M_PI/6.0), 25.0f},
    {11.4f,  8.5f,  static_cast<float>(M_PI/2.5), 25.0f}
};
int wave_index_from_height(float height) {
    for (size_t i = 0; i < waveParamsList.size(); i++) {
        if (std::abs(waveParamsList[i].height - height) < 1e-3f) return int(i);
    }
    return -1;
}

// === Tracker scaffolding copied from your sea_reg pattern ===
AranovskiyFilter<double> arFilter;
KalmANF<double> kalmANF;
FrequencySmoother<float> freqSmoother;
KalmanSmootherVars kalman_freq;
SchmittTriggerFrequencyDetector freqDetector(
    ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS);

static double sim_t = 0.0;
static bool kalm_smoother_first = true;
static uint32_t now_us() { return static_cast<uint32_t>(sim_t * 1e6); }

static void init_tracker_backends() {
    init_filters(&arFilter, &kalman_freq);
    init_filters_alt(&kalmANF, &kalman_freq);
}
static void reset_run_state() {
    sim_t = 0.0;
    kalm_smoother_first = true;
    kalman_smoother_init(&kalman_freq, 0.25f, 2.0f, 100.0f);
    freqSmoother = FrequencySmoother<float>();
    freqDetector.reset();
}
static double run_tracker_once(TrackerType tracker, float a_norm, float dt) {
    double freq = std::numeric_limits<double>::quiet_NaN();
    if (tracker == TrackerType::ARANOVSKIY) {
        freq = estimate_freq(Aranovskiy, &arFilter, &kalmANF,
                             &freqDetector, a_norm, a_norm, dt, now_us());
    } else if (tracker == TrackerType::KALMANF) {
        freq = estimate_freq(Kalm_ANF, &arFilter, &kalmANF,
                             &freqDetector, a_norm, a_norm, dt, now_us());
    } else {
        freq = estimate_freq(ZeroCrossing, &arFilter, &kalmANF,
                             &freqDetector, a_norm, a_norm, dt, now_us());
    }
    float smooth_freq;
    if (kalm_smoother_first) {
        kalm_smoother_first = false;
        freqSmoother.setInitial(freq);
        smooth_freq = float(freq);
    } else {
        smooth_freq = freqSmoother.update(float(freq));
    }
    return smooth_freq;
}

// R_S law based on period Tp (same as your earlier helper)
inline float R_S_law(float Tp, float T_p_base = 8.5f) {
    return R_S_base_global * std::pow(Tp / T_p_base, 1.0f / 3.0f);
}

// ---------------------------
// Per-tracker processing
// ---------------------------
struct OnlineTuneState {
    float tau_applied   = 1.15f;   // s
    float sigma_applied = 1.35f;  // m/s^2
    float RS_applied    =  R_S_law(8.5f); // start from base Tp
};

static void process_wave_file_for_tracker(const std::string &filename,
                                          float dt,
                                          bool with_mag,
                                          TrackerType tracker)
{
    auto parsed = WaveFileNaming::parse_to_params(filename);
    if (!parsed) return;
    auto [kind, type, wp] = *parsed;
    if (kind != FileKind::Data) return;
    if (!(type == WaveType::JONSWAP || type == WaveType::PMSTOKES)) return;

    // Build outname with tracker tag
    std::string trackerName =
        (tracker == TrackerType::ARANOVSKIY) ? "_aran" :
        (tracker == TrackerType::KALMANF)    ? "_kalm" :
                                               "_zero";

    std::string outname = filename;
    auto pos_prefix = outname.find("wave_data_");
    if (pos_prefix != std::string::npos) {
        outname.replace(pos_prefix, std::string("wave_data_").size(), "w3d_");
    } else {
        outname = "w3d_";
    }
    auto pos_ext = outname.rfind(".csv");
    if (pos_ext != std::string::npos) {
        //outname.insert(pos_ext, trackerName);
        outname.insert(pos_ext, with_mag ? "_w3d" : "_w3d_nomag");
    } else {
        outname += (with_mag ? "_w3d" : "_w3d_nomag");
        //outname += trackerName;
        outname += ".csv";
    }

    std::cout << "Processing " << filename
              << " with tracker=" << trackerName
              << " (type=" << EnumTraits<WaveType>::to_string(type) << ")\n";

    // IO
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
        << "gyro_bias_est_x,gyro_bias_est_y,gyro_bias_est_z,"
        << "tau_applied,sigma_a_applied,R_S_applied,"
        << "tau_target,sigma_a_target,R_S_target,"
        << "freq_tracker_hz,Tp_reg_s,accel_var_reg\n";

    // Reader
    WaveDataCSVReader reader(filename);

    // Estimators
    SeaStateRegularity regFilter;  // provides Tp & accel variance & its own internal smoothing for Tp
    OnlineTuneState tune;

    // MEKF (initialize with modest defaults; will adapt slowly)
    const Vector3f sigma_a_init(0.30f, 0.30f, 0.30f);
    const Vector3f sigma_g(0.00134f, 0.00134f, 0.00134f);
    const Vector3f sigma_m(0.3f, 0.3f, 0.3f);
    Kalman3D_Wave<float, true, true> mekf(sigma_a_init, sigma_g, sigma_m);
    mekf.set_aw_time_constant(tune.tau_applied);
    mekf.set_aw_stationary_std(Vector3f::Constant(tune.sigma_applied));
    mekf.set_RS_noise(Vector3f::Constant(tune.RS_applied));

    const Vector3f mag_world_a = MagSim_WMM::mag_world_aero();

    // Noise models
    static NoiseModel accel_noise = make_noise_model(0.03f, 0.02f, 1234);
    static NoiseModel gyro_noise  = make_noise_model(0.001f, 0.0004f, 5678);

    // Adapt alpha per step from fixed per-second rate
    const float alpha_step = 1.0f - std::exp(-ADAPT_RATE_PER_SEC * dt);

    bool first = true;
    bool mag_enabled = false;
    int iter = 0;

    // RMS accumulators (for last 60 s)
    std::vector<float> errs_z, errs_roll, errs_pitch, errs_yaw, errs_angle;

    // Reset tracker internals (sea_reg style)
    reset_run_state();

    // Targets from online estimators
    float tau_target   = NAN;
    float sigma_target = NAN;
    float RS_target    = tune.RS_applied;
    
    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        iter++;

        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);

        // Optional noise
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
                mekf.measurement_update_mag_only(mag_f);  // one-time yaw lock
                mag_enabled = true;
            }
            mekf.measurement_update_acc_only(acc_f);
            if (iter % 3 == 0) {
                mekf.measurement_update_mag_only(mag_f);
            }
        } else {
            mekf.measurement_update_acc_only(acc_f);
        }

        // === Frequency tracking + SeaStateRegularity ===
        // Use noisy vertical accel in Z-up body frame for consistency with your sea_reg pipeline
        float accel_z_noisy = acc_noisy.z();
        float a_norm = accel_z_noisy / g_std;

        // Tracker smoothed frequency (Hz)
        double f_hz = run_tracker_once(tracker, a_norm, dt);
        sim_t = rec.time;  // keep trackers in sync

        // Update regFilter after warmup, using ω = 2π f
        float Tp_reg = NAN;
        float accel_var_reg = NAN;
        if (std::isfinite(f_hz) && rec.time >= ONLINE_TUNE_WARMUP_SEC) {
            regFilter.update(dt, accel_z_noisy, static_cast<float>(2.0 * M_PI * f_hz));
            Tp_reg = regFilter.getDisplacementPeriodSec();
            accel_var_reg = regFilter.getAccelerationVariance();
        }

        if (std::isfinite(f_hz)) {
            float f_clamped = std::clamp(float(f_hz), MIN_FREQ_HZ, MAX_FREQ_HZ);
            tau_target = std::clamp(0.5f / f_clamped, MIN_TAU_S, MAX_TAU_S);
        }
        if (std::isfinite(accel_var_reg)) {
            sigma_target = std::clamp(std::sqrt(std::max(0.0f, accel_var_reg)),
                                      MIN_SIGMA_A, MAX_SIGMA_A);
        }
        if (std::isfinite(Tp_reg)) {
            // Existing law based on Tp (your helper)
            RS_target = R_S_law(Tp_reg);
        }

        // === Slow adaptation (fixed rate) after warmup ===
        if (rec.time >= ONLINE_TUNE_WARMUP_SEC) {
            if (std::isfinite(tau_target)) {
                tune.tau_applied = tune.tau_applied + alpha_step * (tau_target - tune.tau_applied);
                mekf.set_aw_time_constant(tune.tau_applied);
            }
            if (std::isfinite(sigma_target)) {
                tune.sigma_applied = tune.sigma_applied + alpha_step * (sigma_target - tune.sigma_applied);
                mekf.set_aw_stationary_std(Vector3f::Constant(tune.sigma_applied));
            }
            // Always adapt R_S slowly (even if Tp is drifting slowly inside regFilter)
            tune.RS_applied = tune.RS_applied + alpha_step * (RS_target - tune.RS_applied);
            mekf.set_RS_noise(Vector3f::Constant(tune.RS_applied));
        }

        // Estimated quaternion → nautical Euler
        auto coeffs = mekf.quaternion().coeffs();
        Quaternionf q(coeffs(3), coeffs(0), coeffs(1), coeffs(2));
        float r_est_a, p_est_a, y_est_a;
        quat_to_euler_aero(q, r_est_a, p_est_a, y_est_a);
        float r_est = r_est_a, p_est = p_est_a, y_est = y_est_a;
        aero_to_nautical(r_est, p_est, y_est);

        // Reference and estimated kinematics
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

        // Store errors for RMS window
        errs_z.push_back(disp_err.z());
        errs_roll.push_back(r_est - r_ref_out);
        errs_pitch.push_back(p_est - p_ref_out);
        errs_yaw.push_back(y_est - y_ref_out);
        errs_angle.push_back(angle_err);

        // stream row to CSV
        ofs << rec.time << ","
            << r_ref_out << "," << p_ref_out << "," << y_ref_out << ","
            << disp_ref.x() << "," << disp_ref.y() << "," << disp_ref.z() << ","
            << vel_ref.x()  << "," << vel_ref.y()  << "," << vel_ref.z()  << ","
            << acc_ref.x()  << "," << acc_ref.y()  << "," << acc_ref.z()  << ","
            << rec.imu.acc_bx << "," << rec.imu.acc_by << "," << rec.imu.acc_bz << ","
            << rec.imu.gyro_x << "," << rec.imu.gyro_y << "," << rec.imu.gyro_z << ","
            << r_est << "," << p_est << "," << y_est << ","
            << disp_est.x() << "," << disp_est.y() << "," << disp_est.z() << ","
            << vel_est.x()  << "," << vel_est.y()  << "," << vel_est.z() << ","
            << acc_est.x()  << "," << acc_est.y()  << "," << acc_est.z() << ","
            << disp_err.x() << "," << disp_err.y() << "," << disp_err.z() << ","
            << vel_err.x()  << "," << vel_err.y()  << "," << vel_err.z() << ","
            << acc_err.x()  << "," << acc_err.y()  << "," << acc_err.z()  << ","
            << (r_est - r_ref_out) << "," << (p_est - p_ref_out) << "," << (y_est - y_ref_out) << ","
            << angle_err << ","
            << acc_noisy.x() << "," << acc_noisy.y() << "," << acc_noisy.z() << ","
            << gyr_noisy.x() << "," << gyr_noisy.y() << "," << gyr_noisy.z() << ","
            << accel_noise.bias.x() << "," << accel_noise.bias.y() << "," << accel_noise.bias.z() << ","
            << gyro_noise.bias.x()  << "," << gyro_noise.bias.y()  << "," << gyro_noise.bias.z()  << ","
            << bacc_est.x() << "," << bacc_est.y() << "," << bacc_est.z() << ","
            << bgyr_est.x() << "," << bgyr_est.y() << "," << bgyr_est.z() << ","
            << tune.tau_applied << "," << tune.sigma_applied << "," << tune.RS_applied << ","
            << (std::isfinite(tau_target)   ? tau_target   : NAN) << ","
            << (std::isfinite(sigma_target) ? sigma_target : NAN) << ","
            << RS_target << ","
            << (std::isfinite(float(f_hz))  ? float(f_hz)  : NAN) << ","
            << (std::isfinite(Tp_reg)       ? Tp_reg       : NAN) << ","
            << (std::isfinite(accel_var_reg)? accel_var_reg: NAN)
            << "\n";
    });

    ofs.close();
    std::cout << "Wrote " << outname << "\n";

    // === RMS summary for last 60 s ===
    int N_last = static_cast<int>(60.0f / dt);
    if (errs_z.size() > static_cast<size_t>(N_last)) {
        size_t start = errs_z.size() - N_last;

        RMSReport rms_z, rms_roll, rms_pitch, rms_yaw, rms_ang;
        for (size_t i = start; i < errs_z.size(); ++i) {
            rms_z.add(errs_z[i]);
            rms_roll.add(errs_roll[i]);
            rms_pitch.add(errs_pitch[i]);
            rms_yaw.add(errs_yaw[i]);
            rms_ang.add(errs_angle[i]);
        }

        float z_rms = rms_z.rms();
        float z_rms_pct = 100.0f * z_rms / wp.height;

        std::cout << "=== Last 60 s RMS summary for " << outname << " ===\n";
        std::cout << "Z RMS = " << z_rms
                  << " m (" << z_rms_pct
                  << "% of Hs=" << wp.height << ")\n";
        std::cout << "Angles RMS (deg): "
                  << "Roll=" << rms_roll.rms()
                  << ", Pitch=" << rms_pitch.rms()
                  << ", Yaw=" << rms_yaw.rms() << "\n";
        std::cout << "Absolute angle error RMS (deg): "
                  << rms_ang.rms() << "\n";
        std::cout << "tau_target=" << tau_target  << ", sigma_target=" << sigma_target ", RS_target=" << RS_target << "\n";
        std::cout << "=============================================\n\n";

        // FAIL CHECK
        if (type == WaveType::JONSWAP) {
            if (z_rms_pct > FAIL_ERR_LIMIT_PERCENT_LOW) {
                std::cerr << "ERROR: Z RMS above limit (" << z_rms_pct << "%). Failing.\n";
                std::exit(EXIT_FAILURE);
            }
        } else {
            if (z_rms_pct > FAIL_ERR_LIMIT_PERCENT_HIGH) {
                std::cerr << "ERROR: Z RMS above limit (" << z_rms_pct << "%). Failing.\n";
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

// ---------------------------
// Main
// ---------------------------
int main(int argc, char* argv[]) {
    float dt = 1.0f / 240.0f;

    bool with_mag = true;
    add_noise = true;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--nomag") with_mag = false;
        else if (arg == "--no-noise") add_noise = false;
        else if (arg.rfind("--rs-base=", 0) == 0) {
            try { R_S_base_global = std::stof(arg.substr(10)); }
            catch (...) { std::cerr << "Invalid value for --rs-base\n"; return EXIT_FAILURE; }
        }
    }

    init_tracker_backends();

    std::cout << "Simulation starting with_mag=" << (with_mag ? "true" : "false")
              << ", mag_delay=" << MAG_DELAY_SEC << " sec"
              << ", noise=" << (add_noise ? "true" : "false")
              << ", R_S_base=" << R_S_base_global
              << ", adapt_rate=" << ADAPT_RATE_PER_SEC << " /s\n";

    // Gather files
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

    for (const auto& fname : files) {
        // Run all three trackers
        //process_wave_file_for_tracker(fname, dt, with_mag, TrackerType::ARANOVSKIY);
        process_wave_file_for_tracker(fname, dt, with_mag, TrackerType::KALMANF);
        //process_wave_file_for_tracker(fname, dt, with_mag, TrackerType::ZEROCROSS);
    }
    return 0;
}
