#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
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

const float FAIL_ERR_LIMIT_PERCENT_X_HIGH = 50.0f;
const float FAIL_ERR_LIMIT_PERCENT_Y_HIGH = 50.0f;
const float FAIL_ERR_LIMIT_PERCENT_Z_HIGH = 15.0f;

const float FAIL_ERR_LIMIT_PERCENT_X_LOW  = 68.0f;
const float FAIL_ERR_LIMIT_PERCENT_Y_LOW  = 68.0f;
const float FAIL_ERR_LIMIT_PERCENT_Z_LOW  = 15.0f;

// Rolling stats window [s] for RMS and online variance
constexpr float RMS_WINDOW_SEC = 60.0f;

// Online estimation warmup before applying to MEKF [s]
constexpr float ONLINE_TUNE_WARMUP_SEC = 20.0f;

// Adaptation time (seconds)
constexpr float ADAPT_TAU_SEC = 10.0f;
constexpr float ADAPT_EVERY_SECS = 5.0f;

// Stability clamps
constexpr float MIN_SIGMA_A = 0.1f;
constexpr float MAX_SIGMA_A = 20.0f;
constexpr float MIN_FREQ_HZ = 0.1f;
constexpr float MAX_FREQ_HZ = 6.0f;
constexpr float MIN_TAU_S   = 0.1f;
constexpr float MAX_TAU_S   = 11.5f;
constexpr float MIN_R_S     = 0.01f;
constexpr float MAX_R_S     = 20.0f;

// Your project headers
#include "WaveFilesSupport.h"
#include "AranovskiyFilter.h"
#include "KalmANF.h"
#include "FrequencySmoother.h"
#include "KalmanForWaveBasic.h"
#include "KalmanWaveNumStableAlt.h"
#include "SchmittTriggerFrequencyDetector.h"
#include "KalmanSmoother.h"
#include "KalmanWaveDirection.h"
#include "WaveFilters.h"
#include "Kalman3D_Wave.h"
#include "FrameConversions.h"
#include "SeaStateAutoTuner.h"   // <<< switched in

using Eigen::Vector3f;
using Eigen::Quaternionf;

// ---------- RMS helper ----------
class RMSReport {
public:
    inline void add(float value) { sum_sq_ += value * value; count_++; }
    inline float rms() const { return count_ ? std::sqrt(sum_sq_ / float(count_)) : NAN; }
private:
    float sum_sq_ = 0.0f;
    size_t count_ = 0;
};

// ---------- Noise model ----------
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

// ---------- Example waves ----------
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

// ---------- Tracker scaffolding ----------
AranovskiyFilter<double> arFilter;
KalmANF<double> kalmANF;
FrequencySmoother<float> freqSmoother;
KalmanSmootherVars kalman_freq;
SchmittTriggerFrequencyDetector freqDetector(ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS);
static bool kalm_smoother_first = true;
static double sim_t = 0.0;
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
static double clamp_freq(double v) {
    return clamp(v, (double)FREQ_LOWER, (double)FREQ_UPPER);
}
static std::pair<double,bool> run_tracker_once(TrackerType tracker,
                                               float a_norm, float a_raw, float dt) {
    (void)a_raw;
    double freq = FREQ_GUESS;
    if (tracker == TrackerType::ARANOVSKIY)
        freq = estimate_freq(Aranovskiy, &arFilter, &kalmANF, &freqDetector, a_norm, a_norm, dt, now_us());
    else if (tracker == TrackerType::KALMANF)
        freq = estimate_freq(Kalm_ANF, &arFilter, &kalmANF, &freqDetector, a_norm, a_norm, dt, now_us());
    else
        freq = estimate_freq(ZeroCrossing, &arFilter, &kalmANF, &freqDetector, a_norm, a_norm, dt, now_us());
    return {freq, !std::isnan(freq)};
}

// ---------- Online tuning state ----------
struct OnlineTuneState {
    float tau_applied   = 1.15f;              // s
    float sigma_applied = 1.22f;              // m/s²
    float RS_applied    = 8.17f;              // m*s
};

// ---------- main processing ----------
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

    std::string trackerName =
        (tracker == TrackerType::ARANOVSKIY) ? "_aran" :
        (tracker == TrackerType::KALMANF)    ? "_kalm" :
                                               "_zero";

    std::string outname = filename;
    auto pos_prefix = outname.find("wave_data_");
    if (pos_prefix != std::string::npos)
        outname.replace(pos_prefix, std::string("wave_data_").size(), "w3d_");
    else
        outname = "w3d_";
    auto pos_ext = outname.rfind(".csv");
    if (pos_ext != std::string::npos)
        outname.insert(pos_ext, with_mag ? "_w3d" : "_w3d_nomag");
    else {
        outname += (with_mag ? "_w3d" : "_w3d_nomag");
        outname += ".csv";
    }

    std::cout << "Processing " << filename
              << " with tracker=" << trackerName
              << " (type=" << EnumTraits<WaveType>::to_string(type) << ")\n";

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
        << "freq_tracker_hz,Tp_tuner_s,accel_var_tuner\n";

    WaveDataCSVReader reader(filename);

    // ---- Auto Tuner (replaces SeaStateRegularity) ----
    SeaStateAutoTuner tuner;                     // σ_a², EMA f, Tp=1/f, R_S = σ_a / f³
    OnlineTuneState tune;

    // MEKF
    const Vector3f sigma_a_init(0.30f, 0.30f, 0.30f);
    const Vector3f sigma_g(0.00134f, 0.00134f, 0.00134f);
    const Vector3f sigma_m(0.3f, 0.3f, 0.3f);
    Kalman3D_Wave<float, true, true> mekf(sigma_a_init, sigma_g, sigma_m);
    mekf.set_aw_time_constant(tune.tau_applied);
    mekf.set_aw_stationary_corr_std(Vector3f::Constant(tune.sigma_applied));
    mekf.set_RS_noise(Vector3f::Constant(tune.RS_applied));

    const Vector3f mag_world_a = MagSim_WMM::mag_world_aero();

    static NoiseModel accel_noise = make_noise_model(0.03f, 0.02f, 1234);
    static NoiseModel gyro_noise  = make_noise_model(0.001f, 0.0004f, 5678);

    const float alpha_step = 1.0f - std::exp(-dt / ADAPT_TAU_SEC);

    bool first = true;
    bool mag_enabled = false;
    int iter = 0;

    std::vector<float> errs_x, errs_y, errs_z, errs_roll, errs_pitch, errs_yaw, errs_angle;

    reset_run_state();

    // Targets from online estimators
    float tau_target   = NAN;
    float sigma_target = NAN;
    float RS_target    = tune.RS_applied;
    float Tp_reg       = NAN;
    float accel_var_reg = NAN;

    double f_hz = NAN;
    float last_adj = 0.0f;

    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        iter++;

        // Body-frame noisy sensors
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);

        Vector3f acc_noisy = acc_b;
        Vector3f gyr_noisy = gyr_b;
        if (add_noise) {
            acc_noisy = apply_noise(acc_b, accel_noise);
            gyr_noisy = apply_noise(gyr_b, gyro_noise);
        }

        // Transform to NED
        Vector3f acc_f = zu_to_ned(acc_noisy);
        Vector3f gyr_f = zu_to_ned(gyr_noisy);

        // Reference Euler (nautical ENU/Z-up)
        float r_ref_out = rec.imu.roll_deg;
        float p_ref_out = rec.imu.pitch_deg;
        float y_ref_out = rec.imu.yaw_deg;

        // Simulated magnetometer
        Vector3f mag_f(0,0,0);
        if (with_mag) {
            Vector3f mag_b_enu = MagSim_WMM::simulate_mag_from_euler_nautical(r_ref_out, p_ref_out, y_ref_out);
            mag_f = zu_to_ned(mag_b_enu);
        }

        if (first) {
            mekf.initialize_from_acc(acc_f);
            first = false;
        }

        // MEKF updates
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

        // Frequency tracking inputs
        float accel_ref_z   = rec.imu.acc_bz - g_std;
        float accel_z_noisy = accel_ref_z + accel_noise.dist(accel_noise.rng) + accel_noise.bias.z();
        float a_norm        = accel_z_noisy / g_std;

        sim_t = rec.time;  // keep trackers in sync

        // Tracker smoothed frequency (Hz)
        auto [est_freq, updated] = run_tracker_once(tracker, a_norm, accel_z_noisy, dt);
        double smooth_freq = std::numeric_limits<double>::quiet_NaN();
        if (!std::isnan(est_freq) && updated) {
            if (kalm_smoother_first) {
                kalm_smoother_first = false;
                freqSmoother.setInitial(est_freq);
                smooth_freq = est_freq;
            } else {
                smooth_freq = freqSmoother.update(est_freq);
            }
        }
        if (!std::isnan(smooth_freq)) smooth_freq = clamp_freq(smooth_freq);
        f_hz = smooth_freq;

        // ---- SeaStateAutoTuner adaptation ----
        if (std::isfinite(f_hz) && rec.time >= ONLINE_TUNE_WARMUP_SEC) {
            tuner.update(dt, accel_z_noisy, static_cast<float>(f_hz));
            Tp_reg        = tuner.getPeriodSec();      // 1/f
            accel_var_reg = tuner.getAccelVariance();  // σ_a²

            // Targets derived from f and σ_a
            if (std::isfinite(f_hz)) {
                float f_clamped = std::clamp(float(f_hz), MIN_FREQ_HZ, MAX_FREQ_HZ);
                tau_target = std::clamp(0.5f / f_clamped, MIN_TAU_S, MAX_TAU_S);
            }
            if (std::isfinite(accel_var_reg)) {
                sigma_target = std::clamp(std::sqrt(std::max(0.0f, accel_var_reg)), MIN_SIGMA_A, MAX_SIGMA_A);
            }
            if (tuner.isReady()) {
                constexpr float C_adj = 2.0f; // scaling
                RS_target = std::clamp(C_adj * sigma_target * tau_target * tau_target * tau_target, MIN_R_S, MAX_R_S);
            }
        }

        // ---- Slow adaptation (fixed rate) ----
        if (rec.time >= ONLINE_TUNE_WARMUP_SEC) {
            if (std::isfinite(tau_target)) {
                tune.tau_applied += alpha_step * (tau_target - tune.tau_applied);
                if (rec.time - last_adj > ADAPT_EVERY_SECS) {
                    mekf.set_aw_time_constant(tune.tau_applied);
                }
            }
            if (std::isfinite(sigma_target)) {
                tune.sigma_applied += alpha_step * (sigma_target - tune.sigma_applied);
                if (rec.time - last_adj > ADAPT_EVERY_SECS) {
                    mekf.set_aw_stationary_corr_std(Vector3f::Constant(tune.sigma_applied));
                }
            }
            tune.RS_applied += alpha_step * (RS_target - tune.RS_applied);
            if (rec.time - last_adj > ADAPT_EVERY_SECS) {
                mekf.set_RS_noise(Vector3f::Constant(tune.RS_applied));
                last_adj = rec.time;
            }
        }

        // ---- Outputs ----
        // Estimated quaternion -> nautical Euler
        auto coeffs = mekf.quaternion().coeffs(); // (x,y,z,w) in Eigen order
        Quaternionf q(coeffs(3), coeffs(0), coeffs(1), coeffs(2)); // w,x,y,z
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

        Quaternionf q_ref =
            Eigen::AngleAxisf(r_ref_out*M_PI/180.0f, Vector3f::UnitX()) *
            Eigen::AngleAxisf(p_ref_out*M_PI/180.0f, Vector3f::UnitY()) *
            Eigen::AngleAxisf(y_ref_out*M_PI/180.0f, Vector3f::UnitZ());
        Quaternionf q_est_naut =
            Eigen::AngleAxisf(r_est*M_PI/180.0f, Vector3f::UnitX()) *
            Eigen::AngleAxisf(p_est*M_PI/180.0f, Vector3f::UnitY()) *
            Eigen::AngleAxisf(y_est*M_PI/180.0f, Vector3f::UnitZ());
        Quaternionf q_err = q_ref.conjugate() * q_est_naut.normalized();
        float angle_err = 2.0f * std::acos(std::clamp(q_err.w(), -1.0f, 1.0f)) * 180.0f/M_PI;

        // Biases
        Vector3f bacc_est = mekf.get_acc_bias();
        Vector3f bgyr_est = mekf.gyroscope_bias();

        // Append for RMS window
        errs_x.push_back(disp_err.x());
        errs_y.push_back(disp_err.y());
        errs_z.push_back(disp_err.z());
        errs_roll.push_back(r_est - r_ref_out);
        errs_pitch.push_back(p_est - p_ref_out);
        errs_yaw.push_back(y_est - y_ref_out);
        errs_angle.push_back(angle_err);

        // CSV row
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

    // ---------- RMS summary for last 60 s ----------
    int N_last = static_cast<int>(RMS_WINDOW_SEC / dt);
    if (errs_z.size() > static_cast<size_t>(N_last)) {
        size_t start = errs_z.size() - N_last;

        RMSReport rms_x, rms_y, rms_z, rms_roll, rms_pitch, rms_yaw, rms_ang;
        for (size_t i = start; i < errs_z.size(); ++i) {
            rms_x.add(errs_x[i]);
            rms_y.add(errs_y[i]);
            rms_z.add(errs_z[i]);
            rms_roll.add(errs_roll[i]);
            rms_pitch.add(errs_pitch[i]);
            rms_yaw.add(errs_yaw[i]);
            rms_ang.add(errs_angle[i]);
        }

        float x_rms = rms_x.rms();
        float y_rms = rms_y.rms();
        float z_rms = rms_z.rms();

        float x_rms_pct = 100.0f * x_rms / wp.height;
        float y_rms_pct = 100.0f * y_rms / wp.height;
        float z_rms_pct = 100.0f * z_rms / wp.height;

        std::cout << "=== Last 60 s RMS summary for " << outname << " ===\n";
        std::cout << "XYZ RMS (m) = "
                  << "X=" << x_rms << ", Y=" << y_rms << ", Z=" << z_rms << "\n";
        std::cout << "XYZ RMS (% Hs) = "
                  << "X=" << x_rms_pct << "%, Y=" << y_rms_pct << "%, Z=" << z_rms_pct
                  << "% (Hs=" << wp.height << ")\n";
        std::cout << "Angles RMS (deg): "
                  << "Roll=" << rms_roll.rms()
                  << ", Pitch=" << rms_pitch.rms()
                  << ", Yaw=" << rms_yaw.rms() << "\n";
        std::cout << "Absolute angle error RMS (deg): " << rms_ang.rms() << "\n";
        std::cout << "tau_target=" << tau_target  << ", sigma_target=" << sigma_target
                  << ", RS_target=" << RS_target << "\n";
        std::cout << "tau_applied=" << tune.tau_applied  << ", sigma_applied=" << tune.sigma_applied
                  << ", RS_applied=" << tune.RS_applied << "\n";
        std::cout << "f_hz=" << f_hz << ", Tp_tuner=" << Tp_reg << "\n";
        std::cout << "=============================================\n\n";

        // FAIL CHECK (same limits used for X, Y, Z)
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

        fail_if("X", x_rms_pct, limit_x);
        fail_if("Y", y_rms_pct, limit_y);
        fail_if("Z", z_rms_pct, limit_z);
    }
}

// ---------- Main ----------
int main(int argc, char* argv[]) {
    float dt = 1.0f / 240.0f;

    bool with_mag = true;
    add_noise = true;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--nomag") with_mag = false;
        else if (arg == "--no-noise") add_noise = false;
    }

    init_tracker_backends();

    std::cout << "Simulation starting with_mag=" << (with_mag ? "true" : "false")
              << ", mag_delay=" << MAG_DELAY_SEC << " sec"
              << ", noise=" << (add_noise ? "true" : "false")
              << ", adapt_tau_sec=" << ADAPT_TAU_SEC << "\n";

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
        // Run one tracker (enable others if desired)
        process_wave_file_for_tracker(fname, dt, with_mag, TrackerType::KALMANF);
        // process_wave_file_for_tracker(fname, dt, with_mag, TrackerType::ARANOVSKIY);
        // process_wave_file_for_tracker(fname, dt, with_mag, TrackerType::ZEROCROSS);
    }
    return 0;
}
