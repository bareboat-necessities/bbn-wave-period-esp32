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

const float FAIL_ERR_LIMIT_PERCENT_3D_JONSWAP   = 50.0f;
const float FAIL_ERR_LIMIT_PERCENT_3D_PMSTOKES  = 60.0f;

const float FAIL_ERR_LIMIT_PERCENT_Z_JONSWAP   = 14.0f;
const float FAIL_ERR_LIMIT_PERCENT_Z_PMSTOKES  = 16.0f;

const float FAIL_ERR_LIMIT_BIAS_3D_PERCENT = 1600.0f;
const float FAIL_ERR_LIMIT_YAW_DEG = 5.0f;  

constexpr float RMS_WINDOW_SEC = 60.0f;  // RMS window

//  Project headers
#include "WaveFilesSupport.h"
#include "FrameConversions.h"
#include "SeaStateFusionFilter.h"

using Eigen::Vector3f;
using Eigen::Vector2f;

// Direction report helpers
template<typename T>
static T mean_vec(const std::vector<T>& v){
    if (v.empty()) return T(NAN);
    T s = 0; for (auto& x : v) s += x; return s / T(v.size());
}
template<typename T>
static T median_vec(std::vector<T> v){
    if (v.empty()) return T(NAN);
    size_t n = v.size(); std::nth_element(v.begin(), v.begin()+n/2, v.end());
    if (n%2) return v[n/2];
    auto lo = *std::max_element(v.begin(), v.begin()+n/2);
    auto hi = v[n/2];
    return (lo+hi)/T(2);
}
template<typename T>
static T percentile_vec(std::vector<T> v, double p01){
    if (v.empty()) return T(NAN);
    if (p01 <= 0) return *std::min_element(v.begin(), v.end());
    if (p01 >= 1) return *std::max_element(v.begin(), v.end());
    std::sort(v.begin(), v.end());
    double idx = p01 * (v.size()-1);
    size_t i = size_t(std::floor(idx));
    double frac = idx - double(i);
    if (i+1 >= v.size()) return v[i];
    return T(v[i]*(1.0-frac) + v[i+1]*frac);
}
static inline float deg_to_rad(float d){ return d * float(M_PI/180.0); }
static inline float rad_to_deg(float r){ return r * float(180.0/M_PI); }

// Axial angle wrap to [-90, +90] (180° ambiguity)
inline float wrapAxialDeg90(float a) {
    a = std::fmod(a + 180.0f, 360.0f);
    if (a < 0) a += 360.0f;
    a -= 180.0f;                 // [-180,180)
    if (a >  90.0f) a -= 180.0f;  // fold axial
    if (a < -90.0f) a += 180.0f;
    return a;                    // [-90,90]
}

// Generator / nautical-style direction from unit vector:
// 0° = +Y, clockwise positive; then fold axial => [-90,90]
inline float dirDegGeneratorSignedFromVec(const Vector2f& v) {
    // Note: this is atan2(x,y), not atan2(y,x).
    // This converts "math angle from +X CCW" into "heading from +Y CW".
    float deg = rad_to_deg(std::atan2(v.x(), v.y())); // (-180,180]
    return wrapAxialDeg90(deg);
}

struct CircStats {
    float mean_deg = NAN;
    float std_deg  = NAN;
};
static CircStats circular_stats_180(const std::vector<float>& degs){
    CircStats cs;
    if (degs.empty()) return cs;

    // Axial stats: use doubled angles
    double C=0, S=0;
    for (float d : degs){
        const double a2 = 2.0 * deg_to_rad(d);
        C += std::cos(a2);
        S += std::sin(a2);
    }
    C /= double(degs.size());
    S /= double(degs.size());

    const double R = std::sqrt(C*C + S*S);
    const double a2_mean = std::atan2(S, C);

    // Mean for axial data (halve the doubled-angle mean), then wrap to [-90,90]
    float md = float(rad_to_deg(0.5 * a2_mean));
    md = wrapAxialDeg90(md);
    cs.mean_deg = md;

    // Axial circular std (radians): 0.5 * sqrt(-2 ln R)
    cs.std_deg = (R > 1e-12)
        ? float(rad_to_deg(0.5 * std::sqrt(std::max(0.0, -2.0*std::log(R)))))
        : 90.0f;

    // Clamp to the meaningful axial range
    cs.std_deg = std::min(cs.std_deg, 90.0f);
    return cs;
}

static inline const char* wave_dir_to_cstr(WaveDirection w){
    switch (w){
        case FORWARD:  return "TOWARD";
        case BACKWARD: return "AWAY";
        default:       return "UNCERTAIN";
    }
}
static inline int wave_dir_to_num(WaveDirection w){
    switch (w){
        case FORWARD:  return +1;
        case BACKWARD: return -1;
        default:       return 0;
    }
}

inline float wrapDeg(float a) {
    a = std::fmod(a + 180.0f, 360.0f);
    if (a < 0) a += 360.0f;
    return a - 180.0f;
}
inline float diffDeg(float est_deg, float ref_deg) {
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

bool add_noise = true;       //  Noise model
bool attitude_only = false;  //  No OU if true

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
    std::normal_distribution<float> w_uT;   // white noise [uT] per *mag sample*
    std::normal_distribution<float> n01;    // N(0,1) for RW

    Vector3f bias0_uT;      // residual hard-iron after calibration [uT]
    Vector3f bias_rw_uT;    // slowly drifting residual [uT]
    float sigma_bias_rw_uT_sqrt_s = 0.0f;   // [uT]/sqrt(s)

    Eigen::Matrix3f Mis;    // residual soft-iron + misalignment (close to I)
};

MagNoiseModel make_mag_noise_model(
    float sigma_white_uT,          // per mag sample RMS
    float bias_residual_range_uT,  // residual hard-iron half-range (calibrated)
    float sigma_bias_rw_uT_sqrt_s, // slow drift
    float scale_err_max,           // residual scale error (e.g. 0.01 = 1%)
    float cross_axis_max,          // residual cross-axis coupling (e.g. 0.01)
    float misalign_deg_max,        // residual misalignment (deg)
    unsigned seed
){
    MagNoiseModel m;
    m.rng  = std::mt19937(seed);
    m.w_uT = std::normal_distribution<float>(0.0f, sigma_white_uT);
    m.n01  = std::normal_distribution<float>(0.0f, 1.0f);

    // Residual hard-iron (post-calibration, should be small)
    std::uniform_real_distribution<float> ub(-bias_residual_range_uT, bias_residual_range_uT);
    m.bias0_uT = Vector3f(ub(m.rng), ub(m.rng), ub(m.rng));
    m.bias_rw_uT.setZero();
    m.sigma_bias_rw_uT_sqrt_s = sigma_bias_rw_uT_sqrt_s;

    // Residual soft-iron / cross-axis coupling (near identity)
    std::uniform_real_distribution<float> us(1.0f - scale_err_max, 1.0f + scale_err_max);
    std::uniform_real_distribution<float> uc(-cross_axis_max, cross_axis_max);

    Eigen::Matrix3f A = Eigen::Matrix3f::Identity();
    A(0,0) = us(m.rng);
    A(1,1) = us(m.rng);
    A(2,2) = us(m.rng);

    // small symmetric cross-axis terms
    float a01 = uc(m.rng), a02 = uc(m.rng), a12 = uc(m.rng);
    A(0,1) = A(1,0) = a01;
    A(0,2) = A(2,0) = a02;
    A(1,2) = A(2,1) = a12;

    // small misalignment rotation
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

    // Mis = R * A  (still close to I)
    m.Mis = R * A;

    return m;
}

// measurement = Mis * truth + (bias0 + bias_rw) + white
Vector3f apply_mag_noise(const Vector3f& ideal_mag_uT_body, MagNoiseModel& m, float dt_mag)
{
    if (m.sigma_bias_rw_uT_sqrt_s > 0.0f) {
        const float s = m.sigma_bias_rw_uT_sqrt_s * std::sqrt(dt_mag);
        m.bias_rw_uT += Vector3f(s * m.n01(m.rng), s * m.n01(m.rng), s * m.n01(m.rng));
    }
    Vector3f white(m.w_uT(m.rng), m.w_uT(m.rng), m.w_uT(m.rng));
    return (m.Mis * ideal_mag_uT_body) + (m.bias0_uT + m.bias_rw_uT) + white;
}

template <class MekfT>
static inline Vector3f get_mag_bias_est_uT(const MekfT& mekf)
{
    // Try a few likely accessor names. If none exist, return zeros.
    if constexpr (requires { mekf.get_mag_bias_uT(); }) {
        return mekf.get_mag_bias_uT();
    } else if constexpr (requires { mekf.get_mag_bias(); }) {
        return mekf.get_mag_bias();
    } else if constexpr (requires { mekf.magnetometer_bias_uT(); }) {
        return mekf.magnetometer_bias_uT();
    } else if constexpr (requires { mekf.magnetometer_bias(); }) {
        return mekf.magnetometer_bias();
    } else {
        return Vector3f::Zero();
    }
}

//  Main processing
static void process_wave_file_for_tracker(const std::string &filename, float dt, bool with_mag)
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
        outname.insert(pos_ext, with_mag ? "_fusion" : "_fusion_nomag");
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
        << "acc_bias_x,acc_bias_y,acc_bias_z,"                          // TRUE (m/s^2), BODY-NED
        << "gyro_bias_x,gyro_bias_y,gyro_bias_z,"                       // TRUE (rad/s), BODY-NED
        << "acc_bias_est_x,acc_bias_est_y,acc_bias_est_z,"              // EST  (m/s^2), BODY-NED
        << "gyro_bias_est_x,gyro_bias_est_y,gyro_bias_est_z,"           // EST  (rad/s), BODY-NED
        << "mag_bias_x,mag_bias_y,mag_bias_z,"                          // TRUE (uT), BODY-NED
        << "mag_bias_est_x,mag_bias_est_y,mag_bias_est_z,"              // EST  (uT), BODY-NED
        << "mag_bias_err_x,mag_bias_err_y,mag_bias_err_z,"              // EST-TRUE (uT)       
        << "tau_applied,sigma_a_applied,R_S_applied,"
        << "freq_tracker_hz,Tp_tuner_s,accel_var_tuner,"
        << "dir_phase,"
        << "dir_deg,dir_uncert_deg,dir_conf,dir_amp,"
        << "dir_sign,dir_sign_num,"
        << "dir_vec_x,dir_vec_y,"
        << "dfilt_ax,dfilt_ay\n";

    using Fusion = SeaStateFusion<TrackerType::KALMANF>;
    Fusion fusion;

    // Magnetic reference (same each run)
    const Vector3f mag_world_a = MagSim_WMM::mag_world_aero();

    // BMI270-like noise (per-axis), “normal mode” order
    // White noise (per-sample RMS, per axis)
    const float acc_sigma = 1.51e-3f * g_std;  // 1.51 mg-rms -> ~0.0148 m/s^2   [oai_citation:6‡Bosch Sensortec](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi270-ds000.pdf)
    const float gyr_sigma = 0.00157f;         // 0.09 dps-rms -> ~0.00157 rad/s  [oai_citation:7‡Bosch Sensortec](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi270-ds000.pdf)
    
    // “Decently calibrated” residual constant bias half-ranges
    const float acc_bias_range = 5e-3f * g_std;                // 5 mg -> ~0.049 m/s^2
    const float gyr_bias_range = 0.05f * float(M_PI/180.0f);   // 0.05 dps -> ~0.00087 rad/s
    
    // Optional very-slow drift (keep tiny; set 0 if we don’t want drift)
    const float acc_bias_rw = 0.0005f;   // m/s^2 / sqrt(s)
    const float gyr_bias_rw = 0.00001f;  // rad/s / sqrt(s)
    
    ImuNoiseModel accel_noise = make_imu_noise_model(acc_sigma, acc_bias_range, acc_bias_rw, 1234);
    ImuNoiseModel gyro_noise  = make_imu_noise_model(gyr_sigma, gyr_bias_range, gyr_bias_rw, 5678);
    
    // BMM150-like magnetometer behavior (AtomS3R)
    constexpr float MAG_ODR_HZ = 100.0f;                 // 100 (regular) or 20 (high-accuracy)
    constexpr float MAG_DT     = 1.0f / MAG_ODR_HZ;
    // Phase accumulator for exact mag rate
    float mag_phase_s = 0.0f;
    const float mag_sigma_uT = (MAG_ODR_HZ <= 20.0f) ? 0.30f : 0.60f;  // datasheet RMS noise  [oai_citation:6‡Bosch Sensortec](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmm150-ds001.pdf)

    // “Decently calibrated” residuals (small!)
    MagNoiseModel mag_noise = make_mag_noise_model(
        mag_sigma_uT,  // white RMS per mag sample
        2.0f,          // residual hard-iron half-range [uT] (post-cal)
        0.01f,         // slow drift [uT]/sqrt(s)
        0.015f,        // <= 1.5% residual scale
        0.010f,        // <= 1% cross-axis coupling
        1.0f,          // <= 1 deg residual misalignment
        9012
    );
    // Hold-last-sample value (persistent across loop iterations)
    Vector3f mag_body_ned_hold = Vector3f::Zero();
       
    // Filter
    const Vector3f sigma_a_init(2.8f * acc_sigma, 2.8f * acc_sigma, 2.8f * acc_sigma);
    const Vector3f sigma_g(2.0f * gyr_sigma, 2.0f * gyr_sigma, 2.0f * gyr_sigma);    
    const float sigma_m_uT = 1.2f * mag_sigma_uT;   // a bit conservative
    const Vector3f sigma_m(sigma_m_uT, sigma_m_uT, sigma_m_uT);

    Fusion::Config cfg;
    cfg.with_mag = with_mag;
    
    // pass init sigmas
    cfg.sigma_a = sigma_a_init;
    cfg.sigma_g = sigma_g;
    cfg.sigma_m = sigma_m;
    
    // mag ref policy
    cfg.mag_delay_sec = MAG_DELAY_SEC;
    cfg.use_fixed_mag_world_ref = false;   // TODO: switch to false (it needs to learn it by itself)
    cfg.mag_world_ref = mag_world_a;
    
    // warmup policy 
    cfg.freeze_acc_bias_until_live = true;
    cfg.Racc_warmup = 0.5f;
    
    fusion.begin(cfg);
    auto& filter = fusion.raw();
    
    // Optional: attitude-only mode tweaks (via raw() escape hatch)
    if (attitude_only) {
        filter.enableLinearBlock(false);
        filter.mekf().set_initial_acc_bias(Vector3f::Zero());
        filter.mekf().set_initial_acc_bias_std(0.0f);
        filter.mekf().set_Q_bacc_rw(Vector3f::Zero());
        filter.mekf().set_Racc(Vector3f::Constant(0.5f));
    }
    
    WaveDataCSVReader reader(filename);
    
    std::vector<float> errs_x, errs_y, errs_z, errs_roll, errs_pitch, errs_yaw;
    // TRUE displacement history (for 3D ref RMS)
    std::vector<float> ref_x, ref_y, ref_z;    
    // Bias estimation error history (est - true), BODY-NED frame
    std::vector<float> accb_err_x, accb_err_y, accb_err_z;
    std::vector<float> gyrb_err_x, gyrb_err_y, gyrb_err_z;
    std::vector<float> magb_err_x, magb_err_y, magb_err_z;  // [uT]
    // TRUE bias history (for last-window max), BODY-NED frame
    std::vector<float> accb_true_x, accb_true_y, accb_true_z;
    std::vector<float> gyrb_true_x, gyrb_true_y, gyrb_true_z;
    std::vector<float> magb_true_x, magb_true_y, magb_true_z; // [uT]
    // Direction histories
    std::vector<float> freq_hist;
    std::vector<float> dir_deg_hist, dir_unc_hist, dir_conf_hist, dir_amp_hist, dir_phase_hist;
    std::vector<int>   dir_sign_num_hist;
    
    reader.for_each_record([&](const Wave_Data_Sample &rec) {

        // Body-frame raw sensors (Z-up body from CSV)
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);

        if (add_noise) {
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
        if (with_mag) {
            // BMM150-style ODR, independent of whether the filter is using mag yet.
            mag_phase_s += dt;
            bool mag_tick = false;
            if (mag_phase_s >= MAG_DT) {
                while (mag_phase_s >= MAG_DT) mag_phase_s -= MAG_DT;
                mag_tick = true;
            }
            if (mag_tick) {
                Vector3f mag_b_enu = MagSim_WMM::simulate_mag_from_euler_nautical(r_ref_out, p_ref_out, y_ref_out);
                if (add_noise) {
                    mag_b_enu = apply_mag_noise(mag_b_enu, mag_noise, MAG_DT);
                }
                mag_body_ned_hold = zu_to_ned(mag_b_enu);

                // Wrapper decides if/when it’s allowed to use mag
                fusion.updateMag(mag_body_ned_hold);
            }
        }

        // One time update per sample (propagate + accel update)
        fusion.update(dt, gyr_meas_ned, acc_meas_ned, 35.0f);
        
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

        errs_x.push_back(disp_err.x());
        errs_y.push_back(disp_err.y());
        errs_z.push_back(disp_err.z());

        ref_x.push_back(disp_ref.x());
        ref_y.push_back(disp_ref.y());
        ref_z.push_back(disp_ref.z());
        
        errs_roll.push_back(diffDeg(eul_est.x(), r_ref_out));
        errs_pitch.push_back(diffDeg(eul_est.y(), p_ref_out));
        errs_yaw.push_back(diffDeg(eul_est.z(), y_ref_out));
        
        // True biases are generated in the same frame as gyr_b/acc_b (BODY Z-up from CSV)
        Vector3f acc_bias_true_zu  = accel_noise.bias0 + accel_noise.bias_rw;
        Vector3f gyro_bias_true_zu = gyro_noise.bias0  + gyro_noise.bias_rw;
        
        // Convert to the frame used by the filter (BODY-NED), same as acc_meas_ned/gyr_meas_ned
        Vector3f acc_bias_true_ned  = zu_to_ned(acc_bias_true_zu);
        Vector3f gyro_bias_true_ned = zu_to_ned(gyro_bias_true_zu);
        
        // Estimated biases (these are in BODY-NED in the filter pipeline)
        Vector3f acc_bias_est  = filter.mekf().get_acc_bias();
        Vector3f gyro_bias_est = filter.mekf().gyroscope_bias();
        
        // Bias estimation errors (est - true)
        Vector3f acc_bias_err  = acc_bias_est  - acc_bias_true_ned;
        Vector3f gyro_bias_err = gyro_bias_est - gyro_bias_true_ned;

        // True magnetometer additive bias in BODY Z-up (ENU from MagSim), then map to BODY-NED.
        // (This is ONLY the additive hard-iron residual + drift, not Mis.)
        Vector3f mag_bias_true_zu = with_mag
            ? (mag_noise.bias0_uT + mag_noise.bias_rw_uT).eval()
            : Vector3f::Zero().eval();
        Vector3f mag_bias_true_ned = zu_to_ned(mag_bias_true_zu);

        // Estimated magnetometer bias in BODY-NED (uT), if the filter exposes it
        Vector3f mag_bias_est_ned = with_mag ? get_mag_bias_est_uT(filter.mekf()) : Vector3f::Zero();

        // Error (est - true)
        Vector3f mag_bias_err = mag_bias_est_ned - mag_bias_true_ned;

        magb_err_x.push_back(mag_bias_err.x());
        magb_err_y.push_back(mag_bias_err.y());
        magb_err_z.push_back(mag_bias_err.z());
        
        accb_err_x.push_back(acc_bias_err.x());
        accb_err_y.push_back(acc_bias_err.y());
        accb_err_z.push_back(acc_bias_err.z());
        
        gyrb_err_x.push_back(gyro_bias_err.x());
        gyrb_err_y.push_back(gyro_bias_err.y());
        gyrb_err_z.push_back(gyro_bias_err.z());        

        accb_true_x.push_back(acc_bias_true_ned.x());
        accb_true_y.push_back(acc_bias_true_ned.y());
        accb_true_z.push_back(acc_bias_true_ned.z());

        gyrb_true_x.push_back(gyro_bias_true_ned.x());
        gyrb_true_y.push_back(gyro_bias_true_ned.y());
        gyrb_true_z.push_back(gyro_bias_true_ned.z());

        magb_true_x.push_back(mag_bias_true_ned.x());
        magb_true_y.push_back(mag_bias_true_ned.y());
        magb_true_z.push_back(mag_bias_true_ned.z());

        //  Direction telemetry
        const float f_hz = filter.getFreqHz();
        freq_hist.push_back(f_hz);

        auto& d = filter.dir();  // KalmanWaveDirection
        const float dir_phase  = d.getPhase();
        const float dir_unc    = d.getDirectionUncertaintyDegrees();   // ~95% (2σ)
        const float dir_conf   = d.getLastStableConfidence();
        const float dir_amp    = d.getAmplitude();
        const Vector2f dir_vec = d.getDirection();
        const Vector2f dfilt   = d.getFilteredSignal();
        const float dir_deg_gen = dirDegGeneratorSignedFromVec(dir_vec);
        
        // Sense (TOWARD/AWAY) comes from WaveDirectionDetector inside the filter
        WaveDirection sign = UNCERTAIN;
        int sign_num = 0;
        
        // Only report a sign when direction is "good"
        constexpr float CONF_THRESH = 20.0f;
        constexpr float AMP_THRESH  = 0.08f;
        
        if (dir_conf > CONF_THRESH && dir_amp > AMP_THRESH) {
            sign     = filter.getDirSignState();
            sign_num = wave_dir_to_num(sign); // +1 / -1 / 0
        }
        
        const char* sign_str = wave_dir_to_cstr(sign);
        
        // Histories: store the *angle* separately from the *sense*
        dir_phase_hist.push_back(dir_phase);
        dir_deg_hist.push_back(dir_deg_gen);   // <-- THIS is what circular stats should use
        dir_unc_hist.push_back(dir_unc);
        dir_conf_hist.push_back(dir_conf);
        dir_amp_hist.push_back(dir_amp);
        dir_sign_num_hist.push_back(sign_num);

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
            << acc_bias_true_ned.x() << "," << acc_bias_true_ned.y() << "," << acc_bias_true_ned.z() << ","
            << gyro_bias_true_ned.x() << "," << gyro_bias_true_ned.y() << "," << gyro_bias_true_ned.z() << ","
            << acc_bias_est.x()  << "," << acc_bias_est.y()  << "," << acc_bias_est.z()  << ","
            << gyro_bias_est.x() << "," << gyro_bias_est.y() << "," << gyro_bias_est.z() << ","
            << mag_bias_true_ned.x() << "," << mag_bias_true_ned.y() << "," << mag_bias_true_ned.z() << ","
            << mag_bias_est_ned.x()  << "," << mag_bias_est_ned.y()  << "," << mag_bias_est_ned.z()  << ","
            << mag_bias_err.x()      << "," << mag_bias_err.y()      << "," << mag_bias_err.z()      << ","           
            << filter.getTauApplied() << ","
            << filter.getSigmaApplied() << ","
            << filter.getRSApplied() << ","
            << filter.getFreqHz() << ","
            << filter.getPeriodSec() << ","
            << filter.getAccelVariance() << ","
            << dir_phase << "," << dir_deg_gen << "," << dir_unc << "," << dir_conf  << "," << dir_amp << ","
            << sign_str << "," << sign_num << "," << dir_vec.x()  << "," << dir_vec.y()  << "," << dfilt.x()  << ","
            << dfilt.y() << "\n";
    });
    
    ofs.close();
    std::cout << "Wrote " << outname << "\n";

    //  RMS summary (last 60 s)
    int N_last = static_cast<int>(RMS_WINDOW_SEC / dt);
    if (errs_z.size() > static_cast<size_t>(N_last)) {
        size_t start = errs_z.size() - N_last;

        RMSReport rms_x, rms_y, rms_z, rms_roll, rms_pitch, rms_yaw;
        RMSReport rms_accb_x, rms_accb_y, rms_accb_z;
        RMSReport rms_gyrb_x, rms_gyrb_y, rms_gyrb_z;
        RMSReport rms_magb_x, rms_magb_y, rms_magb_z;

        float acc_true_max_x = 0.f, acc_true_max_y = 0.f, acc_true_max_z = 0.f, acc_true_max_3d = 0.f;
        float gyr_true_max_x = 0.f, gyr_true_max_y = 0.f, gyr_true_max_z = 0.f, gyr_true_max_3d = 0.f;
        float mag_true_max_x = 0.f, mag_true_max_y = 0.f, mag_true_max_z = 0.f, mag_true_max_3d = 0.f;
        float disp_true_max_3d = 0.f;
        
        for (size_t i = start; i < errs_z.size(); ++i) {
            rms_x.add(errs_x[i]);
            rms_y.add(errs_y[i]);
            rms_z.add(errs_z[i]);

            // Max TRUE 3D displacement amplitude in window
            const float dx = ref_x[i];
            const float dy = ref_y[i];
            const float dz = ref_z[i];
            const float r  = std::sqrt(dx*dx + dy*dy + dz*dz);
            disp_true_max_3d = std::max(disp_true_max_3d, r);
            
            rms_roll.add(errs_roll[i]);
            rms_pitch.add(errs_pitch[i]);
            rms_yaw.add(errs_yaw[i]);
        
            rms_accb_x.add(accb_err_x[i]);
            rms_accb_y.add(accb_err_y[i]);
            rms_accb_z.add(accb_err_z[i]);
        
            rms_gyrb_x.add(gyrb_err_x[i]);
            rms_gyrb_y.add(gyrb_err_y[i]);
            rms_gyrb_z.add(gyrb_err_z[i]);

            rms_magb_x.add(magb_err_x[i]);
            rms_magb_y.add(magb_err_y[i]);
            rms_magb_z.add(magb_err_z[i]);

            // max TRUE bias in the same window (per-axis abs max + max vector norm)
            {
                const float ax = accb_true_x[i], ay = accb_true_y[i], az = accb_true_z[i];
                acc_true_max_x = std::max(acc_true_max_x, std::abs(ax));
                acc_true_max_y = std::max(acc_true_max_y, std::abs(ay));
                acc_true_max_z = std::max(acc_true_max_z, std::abs(az));
                acc_true_max_3d = std::max(acc_true_max_3d, std::sqrt(ax*ax + ay*ay + az*az));

                const float gx = gyrb_true_x[i], gy = gyrb_true_y[i], gz = gyrb_true_z[i];
                gyr_true_max_x = std::max(gyr_true_max_x, std::abs(gx));
                gyr_true_max_y = std::max(gyr_true_max_y, std::abs(gy));
                gyr_true_max_z = std::max(gyr_true_max_z, std::abs(gz));
                gyr_true_max_3d = std::max(gyr_true_max_3d, std::sqrt(gx*gx + gy*gy + gz*gz));

                const float mx = magb_true_x[i], my = magb_true_y[i], mz = magb_true_z[i];
                mag_true_max_x = std::max(mag_true_max_x, std::abs(mx));
                mag_true_max_y = std::max(mag_true_max_y, std::abs(my));
                mag_true_max_z = std::max(mag_true_max_z, std::abs(mz));
                mag_true_max_3d = std::max(mag_true_max_3d, std::sqrt(mx*mx + my*my + mz*mz));
            }           
        }

        float x_rms = rms_x.rms(), y_rms = rms_y.rms(), z_rms = rms_z.rms();
        
        // Per-axis % still relative to Hs:
        float x_pct = 100.f * x_rms / wp.height;
        float y_pct = 100.f * y_rms / wp.height;
        float z_pct = 100.f * z_rms / wp.height;

        // 3D RMS(error)
        float rms_3d_err = std::sqrt(x_rms * x_rms + y_rms * y_rms + z_rms * z_rms);

        // 3D error as % of max TRUE 3D displacement amplitude in the same window
        float pct_3d = (disp_true_max_3d > 1e-12f && std::isfinite(rms_3d_err))
                         ? 100.f * rms_3d_err / disp_true_max_3d
                         : NAN;

        std::cout << "=== Last 60 s RMS summary for " << outname << " ===\n";
        std::cout << "XYZ RMS (m): X=" << x_rms << " Y=" << y_rms << " Z=" << z_rms << "\n";
        std::cout << "XYZ RMS (%Hs): X=" << x_pct << "% Y=" << y_pct << "% Z=" << z_pct << "% (Hs=" << wp.height << ")\n";
        std::cout << "3D RMS (m): " << rms_3d_err
                  << " (3D % of max |disp_ref|_3D = " << pct_3d << "%, max |disp_ref|_3D = "
                  << disp_true_max_3d << " m)\n";
        std::cout << "Angles RMS (deg): Roll=" << rms_roll.rms()
                  << " Pitch=" << rms_pitch.rms()
                  << " Yaw=" << rms_yaw.rms() << "\n";
        
        // Bias error RMS (vector RMS = sqrt(mean(||e||^2)) = sqrt(rms_x^2 + rms_y^2 + rms_z^2))
        auto vec_rms = [](float rx, float ry, float rz) {
            return std::sqrt(rx*rx + ry*ry + rz*rz);
        };
        const float accb_rx = rms_accb_x.rms(), accb_ry = rms_accb_y.rms(), accb_rz = rms_accb_z.rms();
        const float gyrb_rx = rms_gyrb_x.rms(), gyrb_ry = rms_gyrb_y.rms(), gyrb_rz = rms_gyrb_z.rms();
        const float accb_r3 = vec_rms(accb_rx, accb_ry, accb_rz);
        const float gyrb_r3 = vec_rms(gyrb_rx, gyrb_ry, gyrb_rz);
        
        std::cout << "Bias error RMS (acc, m/s^2): "
                  << "X=" << accb_rx << " Y=" << accb_ry << " Z=" << accb_rz
                  << " |3D|=" << accb_r3 << "\n";
        std::cout << "Bias error RMS (gyro, rad/s): "
                  << "X=" << gyrb_rx << " Y=" << gyrb_ry << " Z=" << gyrb_rz
                  << " |3D|=" << gyrb_r3 << "\n";
        // gyro bias RMS also in deg/s for readability
        const float rad2deg = 180.0f / float(M_PI);
        std::cout << "Bias error RMS (gyro, deg/s): "
                  << "X=" << (gyrb_rx*rad2deg) << " Y=" << (gyrb_ry*rad2deg) << " Z=" << (gyrb_rz*rad2deg)
                  << " |3D|=" << (gyrb_r3*rad2deg) << "\n";

        const float magb_rx = rms_magb_x.rms(), magb_ry = rms_magb_y.rms(), magb_rz = rms_magb_z.rms();
        const float magb_r3 = vec_rms(magb_rx, magb_ry, magb_rz);
        std::cout << "Bias error RMS (mag, uT): "
                  << "X=" << magb_rx << " Y=" << magb_ry << " Z=" << magb_rz
                  << " |3D|=" << magb_r3 << "\n";        

        auto pct_of_max = [](float rms, float maxv) -> float {
            return (maxv > 1e-12f && std::isfinite(rms)) ? (100.f * rms / maxv) : NAN;
        };
        
        // Print max TRUE bias used for normalization
        std::cout << "Max TRUE bias in window (acc, m/s^2): "
                  << "X=" << acc_true_max_x << " Y=" << acc_true_max_y << " Z=" << acc_true_max_z
                  << " |3D|=" << acc_true_max_3d << "\n";
        std::cout << "Max TRUE bias in window (gyro, rad/s): "
                  << "X=" << gyr_true_max_x << " Y=" << gyr_true_max_y << " Z=" << gyr_true_max_z
                  << " |3D|=" << gyr_true_max_3d << "\n";
        std::cout << "Max TRUE bias in window (mag, uT): "
                  << "X=" << mag_true_max_x << " Y=" << mag_true_max_y << " Z=" << mag_true_max_z
                  << " |3D|=" << mag_true_max_3d << "\n";
        
        // cache 3D percentages for bias error
        float accb_r3_pct = pct_of_max(accb_r3, acc_true_max_3d);
        float gyrb_r3_pct = pct_of_max(gyrb_r3, gyr_true_max_3d);
        float magb_r3_pct = pct_of_max(magb_r3, mag_true_max_3d); // optional, for diagnostics
        
        // Print bias error RMS as % of max TRUE bias (same window)
        std::cout << "Bias error RMS (% of max TRUE bias) (acc): "
                  << "X=" << pct_of_max(accb_rx, acc_true_max_x) << "% "
                  << "Y=" << pct_of_max(accb_ry, acc_true_max_y) << "% "
                  << "Z=" << pct_of_max(accb_rz, acc_true_max_z) << "% "
                  << "|3D|=" << accb_r3_pct << "%\n";
        std::cout << "Bias error RMS (% of max TRUE bias) (gyro): "
                  << "X=" << pct_of_max(gyrb_rx, gyr_true_max_x) << "% "
                  << "Y=" << pct_of_max(gyrb_ry, gyr_true_max_y) << "% "
                  << "Z=" << pct_of_max(gyrb_rz, gyr_true_max_z) << "% "
                  << "|3D|=" << gyrb_r3_pct << "%\n";
        std::cout << "Bias error RMS (% of max TRUE bias) (mag): "
                  << "X=" << pct_of_max(magb_rx, mag_true_max_x) << "% "
                  << "Y=" << pct_of_max(magb_ry, mag_true_max_y) << "% "
                  << "Z=" << pct_of_max(magb_rz, mag_true_max_z) << "% "
                  << "|3D|=" << magb_r3_pct << "%\n";

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

        // Direction Report
        {
            const size_t i0 = start;
            const size_t i1 = errs_z.size();

            if (i0 < i1 && i1 <= dir_deg_hist.size()) {
                std::vector<float> vf(freq_hist.begin()+i0,       freq_hist.begin()+i1);
                std::vector<float> vd(dir_deg_hist.begin()+i0,    dir_deg_hist.begin()+i1);
                std::vector<float> vu(dir_unc_hist.begin()+i0,    dir_unc_hist.begin()+i1);
                std::vector<float> vc(dir_conf_hist.begin()+i0,   dir_conf_hist.begin()+i1);
                std::vector<float> va(dir_amp_hist.begin()+i0,    dir_amp_hist.begin()+i1);
                
                // Drop NaNs/Infs from vd so circular stats can't become NaN
                vd.erase(std::remove_if(vd.begin(), vd.end(),
                                        [](float a){ return !std::isfinite(a); }),
                         vd.end());
                
                auto cs = circular_stats_180(vd);

                // Count sign states + "good" samples in this window
                int nToward = 0, nAway = 0, nUnc = 0;
                size_t good = 0;
                constexpr float CONF_THRESH = 20.0f;
                constexpr float AMP_THRESH  = 0.08f;
                
                for (size_t k = i0; k < i1; ++k) {
                    const int s = dir_sign_num_hist[k];
                    if (s > 0) ++nToward;
                    else if (s < 0) ++nAway;
                    else ++nUnc;
                
                    if (dir_conf_hist[k] > CONF_THRESH && dir_amp_hist[k] > AMP_THRESH)
                        ++good;
                }                
                const int nWin = int(i1 - i0);
                auto pct = [&](int n){ return (nWin > 0) ? (100.0 * double(n) / double(nWin)) : 0.0; };

                std::cout << "=== Direction Report (last 60 s only) for " << outname << " ===\n";
                std::cout << "window_s: " << (float(i1 - i0) * dt)
                          << " samples: " << (i1 - i0) << "\n";
                std::cout << "freq_hz: mean=" << mean_vec(vf)
                          << " median=" << median_vec(vf)
                          << " p05=" << percentile_vec(vf,0.05)
                          << " p95=" << percentile_vec(vf,0.95) << "\n";
                std::cout << "dir_deg_gen ([-90,90], 0=+Y CW): mean_circ=" << cs.mean_deg
                          << " circ_std≈" << cs.std_deg << " deg\n";
                std::cout << "uncert_deg: mean=" << mean_vec(vu)
                          << " median=" << median_vec(vu)
                          << " p95=" << percentile_vec(vu,0.95) << "\n";
                std::cout << "confidence: mean=" << mean_vec(vc)
                          << " >" << CONF_THRESH << " count=" << good
                          << " (" << (100.0 * double(good)/double(i1-i0)) << "%)\n";
                std::cout << "sign: TOWARD=" << nToward << " (" << pct(nToward) << "%)"
                          << " AWAY=" << nAway << " (" << pct(nAway) << "%)"
                          << " UNCERTAIN=" << nUnc << " (" << pct(nUnc) << "%)\n";
                std::cout << "=============================================\n\n";
            }
        }

        // Failure criteria 
        float limit_z  = (type == WaveType::JONSWAP) ? FAIL_ERR_LIMIT_PERCENT_Z_JONSWAP : FAIL_ERR_LIMIT_PERCENT_Z_PMSTOKES;
        float limit_3d = (type == WaveType::JONSWAP) ? FAIL_ERR_LIMIT_PERCENT_3D_JONSWAP : FAIL_ERR_LIMIT_PERCENT_3D_PMSTOKES;

        auto fail_if = [&](const char* label, float pct, float limit) {
            if (pct > limit) {
                std::cerr << "ERROR: " << label << " RMS above limit ("<< pct << "% > " << limit << "%). Failing.\n";
                std::exit(EXIT_FAILURE);
            }
        };
  
        fail_if("Z",  z_pct,  limit_z);   // Keep tight vertical gate
        fail_if("3D", pct_3d, limit_3d);  // 3D displacement gate instead of X/Y axis gates

        if (rms_yaw.rms() > FAIL_ERR_LIMIT_YAW_DEG) {
            std::cerr << "ERROR: Yaw RMS above limit ("
                      << rms_yaw.rms() << " deg > " << FAIL_ERR_LIMIT_YAW_DEG
                      << " deg). Failing.\n";
            std::exit(EXIT_FAILURE);
        }
        // 3D bias error fail gates at % of max TRUE bias
        if (std::isfinite(accb_r3_pct) && accb_r3_pct > FAIL_ERR_LIMIT_BIAS_3D_PERCENT) {
            std::cerr << "ERROR: 3D accel bias error RMS above limit ("
                      << accb_r3_pct << "% > " << FAIL_ERR_LIMIT_BIAS_3D_PERCENT
                      << "% of max TRUE bias). Failing.\n";
            std::exit(EXIT_FAILURE);
        }        
        if (std::isfinite(gyrb_r3_pct) && gyrb_r3_pct > FAIL_ERR_LIMIT_BIAS_3D_PERCENT) {
            std::cerr << "ERROR: 3D gyro bias error RMS above limit ("
                      << gyrb_r3_pct << "% > " << FAIL_ERR_LIMIT_BIAS_3D_PERCENT
                      << "% of max TRUE bias). Failing.\n";
            std::exit(EXIT_FAILURE);
        }        
    }
}

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
