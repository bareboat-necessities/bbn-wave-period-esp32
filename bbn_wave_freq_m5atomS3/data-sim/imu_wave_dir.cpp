/*
    Copyright (c) 2025
    Mikhail Grushinskiy
*/

#define EIGEN_NON_ARDUINO

#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <random>
#include <memory>
#include <cmath>

#include <Eigen/Dense>

using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Matrix2f;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float g_std = 9.80665f;     // standard gravity acceleration m/s²

#include "WaveFilesSupport.h"
#include "FrameConversions.h"
#include "AranovskiyFilter.h"
#include "KalmANF.h"
#include "SchmittTriggerFrequencyDetector.h"
#include "FrequencySmoother.h"

#define ZERO_CROSSINGS_SCALE          1.0f
#define ZERO_CROSSINGS_DEBOUNCE_TIME  0.12f
#define ZERO_CROSSINGS_STEEPNESS_TIME 0.21f

#define FREQ_GUESS 0.3f   // Hz

// CLI & sim flags
static bool add_noise = true;

struct NoiseModel {
    std::mt19937 rng;
    std::normal_distribution<float> dist;
    Vector3f bias;
};
NoiseModel make_noise_model(float sigma, float bias_range, unsigned seed) {
    NoiseModel m{std::mt19937(seed),
                 std::normal_distribution<float>(0.0f, sigma),
                 Vector3f::Zero()};
    std::uniform_real_distribution<float> ub(-bias_range, bias_range);
    m.bias = Vector3f(ub(m.rng), ub(m.rng), ub(m.rng));
    return m;
}
Vector3f apply_noise(const Vector3f& v, NoiseModel& m) {
    return v - m.bias + Vector3f(m.dist(m.rng), m.dist(m.rng), m.dist(m.rng));
}

class EIGEN_ALIGN_MAX KalmanWaveDirection {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KalmanWaveDirection(float initialOmega, float deltaT)
        : omega(initialOmega), phase(0.0f) {
        reset(deltaT);
    }

    void reset(float /*deltaT*/) {
        A_est.setZero();
        P = Matrix2f::Identity() * 1.0f;
        confidence = 0.0f;
        lastStableConfidence = 0.0f;
        lastStableCovariance = Matrix2f::Identity();
        lastStableDir = Vector2f(1.0f, 0.0f);
        lastStableAmplitude = 0.0f;
    }

    void update(float ax, float ay, float currentOmega, float deltaT) {
        if (std::fabs(currentOmega - omega) > 0.01f * std::fabs(omega)) {
            omega = currentOmega;
        }
        // Advance phase
        updatePhase(deltaT);

        const float c = std::cos(phase);
        if (std::fabs(c) < 0.001f) {
            P += Q;
            P = 0.5f * (P + P.transpose());
            confidence = 1.0f / (P.trace() + 1e-6f);
            confidence *= 0.98f;
            return;
        }
        Matrix2f H = c * Matrix2f::Identity();

        // Predict
        Vector2f A_pred = A_est;
        Matrix2f P_pred = P + Q;

        // Regularize
        P_pred = 0.5f * (P_pred + P_pred.transpose());
        P_pred += Matrix2f::Identity() * 1e-10f;

        // Kalman gain
        Matrix2f S = H * P_pred * H.transpose() + R;
        Matrix2f K = P_pred * H.transpose() * S.ldlt().solve(Matrix2f::Identity());

        // Measurement
        Vector2f z(ax, ay);

        // Update state
        A_est = A_pred + K * (z - H * A_pred);

        // Joseph covariance update
        Matrix2f I = Matrix2f::Identity();
        Matrix2f KH = K * H;
        P = (I - KH) * P_pred * (I - KH).transpose() + K * R * K.transpose();
        P = 0.5f * (P + P.transpose());

        confidence = 1.0f / (P.trace() + 1e-6f);
    }

    Vector2f getDirection() const {
        float norm = A_est.norm();
        const float AMP_THRESHOLD = 0.08f;
        const float CONFIDENCE_THRESHOLD = 20.0f;
        if (norm > AMP_THRESHOLD && confidence > CONFIDENCE_THRESHOLD) {
            Vector2f newDir = A_est / norm;
            if (lastStableDir.dot(newDir) < 0.0f) newDir = -newDir;
            const float alpha = 0.05f;
            lastStableDir = ((1.0f - alpha) * lastStableDir + alpha * newDir).normalized();
            lastStableAmplitude = norm;
            lastStableConfidence = confidence;
            lastStableCovariance = P;
        }
        return lastStableDir;
    }

    float getDirectionDegrees() const {
        Vector2f dir = getDirection();
        float deg = std::atan2(dir.y(), dir.x()) * (180.0f / float(M_PI));
        if (deg < 0.0f) deg += 180.0f;
        if (deg >= 180.0f) deg -= 180.0f;
        return deg;
    }

    float getDirectionUncertaintyDegrees() const {
        const float amp = lastStableAmplitude;
        if (amp < 1e-6f) return 180.0f;
        Vector2f dir = A_est / amp;
        Vector2f tangent(-dir.y(), dir.x());
        float angular_var = tangent.transpose() * lastStableCovariance * tangent;
        float angular_std_rad = std::sqrt(std::max(0.0f, angular_var)) / std::max(1e-6f, amp);
        float angle_deg = (2.0f * angular_std_rad) * (180.0f / float(M_PI));
        return std::max(0.0f, std::min(angle_deg, 180.0f));
    }

    float getConfidence() const { return confidence; }

    Vector2f getFilteredSignal() const {
        return A_est * std::cos(phase) + Vector2f(-A_est.y(), A_est.x()) * std::sin(phase);
    }
    Vector2f getOscillationAlongDirection() const { return A_est * std::cos(phase); }
    Vector2f getAmplitudeVector() const { return A_est; }
    float getAmplitude() const { return A_est.norm(); }
    float getPhase() const { return phase; }

    void setProcessNoise(float q) { Q = Matrix2f::Identity() * q; }
    void setMeasurementNoise(float r) { R = Matrix2f::Identity() * r; }

private:
    void updatePhase(float deltaT) {
        phase = std::remainder(phase + omega * deltaT, 2.0f * float(M_PI));
    }

    // State
    Vector2f A_est = Vector2f::Zero();
    Matrix2f P = Matrix2f::Identity();
    Matrix2f Q = Matrix2f::Identity() * 1e-6f;
    Matrix2f R = Matrix2f::Identity() * 0.01f;

    float omega = 2.0f * float(M_PI) * FREQ_GUESS; // rad/s
    float phase = 0.0f;
    float confidence = 0.0f;

    // Stable (smoothed) info
    mutable Vector2f  lastStableDir = Vector2f(1.0f, 0.0f);
    mutable float     lastStableAmplitude = 0.0f;
    mutable float     lastStableConfidence = 0.0f;
    mutable Matrix2f  lastStableCovariance = Matrix2f::Identity();
};

template<TrackerType> struct TrackerPolicy; // fwd

// Aranovskiy
template<> struct TrackerPolicy<TrackerType::ARANOVSKIY> {
    using Tracker = AranovskiyFilter<double>;
    Tracker t;
    TrackerPolicy() {
        const double omega_up   = (FREQ_GUESS * 2.0) * (2.0 * M_PI);
        const double k_gain     = 20.0;
        const double x1_0       = 0.0;
        const double omega_init = (FREQ_GUESS / 1.5) * 2.0 * M_PI;
        const double theta_0    = -(omega_init * omega_init);
        const double sigma_0    = theta_0;
        t.setParams(omega_up, k_gain);
        t.setState(x1_0, theta_0, sigma_0);
    }
    inline double run(float a_vert_inertial, float dt) {
        t.update(double(a_vert_inertial) / double(g_std), double(dt));
        return t.getFrequencyHz();
    }
};

// KalmANF
template<> struct TrackerPolicy<TrackerType::KALMANF> {
    using Tracker = KalmANF<double>;
    Tracker t = Tracker();
    inline double run(float a_vert_inertial, float dt) {
        double e;
        return t.process(double(a_vert_inertial) / double(g_std), double(dt), &e);
    }
};

// ZeroCross
#define ZERO_CROSSINGS_HYSTERESIS 0.04f
#define ZERO_CROSSINGS_PERIODS    1
template<> struct TrackerPolicy<TrackerType::ZEROCROSS> {
    using Tracker = SchmittTriggerFrequencyDetector;
    Tracker t = Tracker(ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS);
    inline double run(float a_vert_inertial, float dt) {
        const float f = t.update(a_vert_inertial / g_std,
                                 ZERO_CROSSINGS_SCALE,
                                 ZERO_CROSSINGS_DEBOUNCE_TIME,
                                 ZERO_CROSSINGS_STEEPNESS_TIME,
                                 dt);
        if (f == SCHMITT_TRIGGER_FREQ_INIT || f == SCHMITT_TRIGGER_FALLBACK_FREQ) return FREQ_GUESS;
        return f;
    }
};

constexpr float MIN_FREQ_HZ = 0.1f;
constexpr float MAX_FREQ_HZ = 5.0f;

static inline float deg_to_rad(float d){ return d * float(M_PI/180.0); }
static inline float rad_to_deg(float r){ return r * float(180.0/M_PI); }

template<typename T>
static T mean(const std::vector<T>& v){
    if (v.empty()) return T(NAN);
    T s = 0; for (auto& x : v) s += x; return s / T(v.size());
}
template<typename T>
static T median(std::vector<T> v){
    if (v.empty()) return T(NAN);
    size_t n = v.size(); std::nth_element(v.begin(), v.begin()+n/2, v.end());
    if (n%2) return v[n/2];
    auto lo = *std::max_element(v.begin(), v.begin()+n/2);
    auto hi = v[n/2];
    return (lo+hi)/T(2);
}
template<typename T>
static T percentile(std::vector<T> v, double p01){  // p01 in [0,1]
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

// Circular mean/std for directions on [0,180):
// Map θ → 2θ on circle, average, then halve result.
struct CircStats {
    float mean_deg = NAN;
    float std_deg  = NAN;  // approx von Mises std (2σ-ish feel)
};
static CircStats circular_stats_180(const std::vector<float>& degs){
    CircStats cs;
    if (degs.empty()) return cs;
    double C=0, S=0;
    for (float d : degs){
        double a2 = 2.0 * deg_to_rad(d);
        C += std::cos(a2);
        S += std::sin(a2);
    }
    C /= double(degs.size());
    S /= double(degs.size());
    double R = std::sqrt(C*C + S*S);
    double a2_mean = std::atan2(S, C);           // in [-π, π]
    double a_mean  = 0.5 * a2_mean;              // halve back to 180° space
    float md = float(rad_to_deg(a_mean));
    if (md < 0) md += 180.0f;                    // wrap to [0,180)
    cs.mean_deg = md;

    // circular std for doubled angles, then halve
    // sigma = sqrt(-2 ln R) / 2  (in radians) → deg
    if (R > 1e-9) {
        cs.std_deg = float(rad_to_deg(0.5 * std::sqrt(std::max(0.0, -2.0*std::log(R)))));
    } else {
        cs.std_deg = 90.0f;
    }
    return cs;
}

template<TrackerType T>
class WaveDirectionEstimator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    WaveDirectionEstimator(float dt,
                           float f_init_hz = FREQ_GUESS,
                           float R_meas    = 0.03f*0.03f,
                           float Q_proc    = 1e-6f)
    : dt_(dt)
    , f_smoother_()
    , freq_hz_(f_init_hz)
    , dirFilter_(2.0f * float(M_PI) * f_init_hz, dt)
    {
        dirFilter_.setMeasurementNoise(R_meas);
        dirFilter_.setProcessNoise(Q_proc);
        f_smoother_.setInitial(f_init_hz);
    }

    inline void update(const Vector3f& acc_body_ned) {
        const float a_z_inertial = acc_body_ned.z() + g_std;  // vertical inertial accel
        double f_raw = tracker_.run(a_z_inertial, dt_);
        float f_clamped = std::min(std::max(float(f_raw), MIN_FREQ_HZ), MAX_FREQ_HZ);
        freq_hz_ = f_smoother_.update(f_clamped);

        const float omega = 2.0f * float(M_PI) * freq_hz_;
        dirFilter_.update(acc_body_ned.x(), acc_body_ned.y(), omega, dt_);
    }

    // Accessors
    inline float        getFrequencyHz() const { return freq_hz_; }
    inline float        getDirectionDegrees() const { return dirFilter_.getDirectionDegrees(); }
    inline float        getDirectionUncertaintyDegrees() const { return dirFilter_.getDirectionUncertaintyDegrees(); }
    inline float        getConfidence() const { return dirFilter_.getConfidence(); }
    inline float        getAmplitude() const { return dirFilter_.getAmplitude(); }
    inline Vector2f     getDirectionUnit() const { return dirFilter_.getDirection(); }
    inline Vector2f     getFilteredXY() const { return dirFilter_.getFilteredSignal(); }
    inline float        getPhase() const { return dirFilter_.getPhase(); }

    inline void setR(float r) { dirFilter_.setMeasurementNoise(r); }
    inline void setQ(float q) { dirFilter_.setProcessNoise(q); }

private:
    float dt_;
    TrackerPolicy<T> tracker_;
    FrequencySmoother<float> f_smoother_;
    float freq_hz_;
    KalmanWaveDirection dirFilter_;
};

// Runtime interface
struct IWaveDir {
    virtual ~IWaveDir() = default;
    virtual void   update(const Vector3f& acc_body_ned) = 0;
    virtual float  getFrequencyHz() const = 0;
    virtual float  getDirectionDegrees() const = 0;
    virtual float  getDirectionUncertaintyDegrees() const = 0;
    virtual float  getConfidence() const = 0;
    virtual float  getAmplitude() const = 0;
    virtual Vector2f getDirectionUnit() const = 0;
    virtual Vector2f getFilteredXY() const = 0;
    virtual float    getPhase() const = 0;
};

template<TrackerType T>
struct WaveDirWrap : IWaveDir {
    WaveDirectionEstimator<T> est;
    explicit WaveDirWrap(float dt) : est(dt) {}
    void update(const Vector3f& a) override { est.update(a); }
    float  getFrequencyHz() const override { return est.getFrequencyHz(); }
    float  getDirectionDegrees() const override { return est.getDirectionDegrees(); }
    float  getDirectionUncertaintyDegrees() const override { return est.getDirectionUncertaintyDegrees(); }
    float  getConfidence() const override { return est.getConfidence(); }
    float  getAmplitude() const override { return est.getAmplitude(); }
    Vector2f getDirectionUnit() const override { return est.getDirectionUnit(); }
    Vector2f getFilteredXY() const override { return est.getFilteredXY(); }
    float    getPhase() const override { return est.getPhase(); }
};

static std::unique_ptr<IWaveDir> make_dir_estimator(const std::string& name, float dt) {
    if (name == "aran")  return std::make_unique<WaveDirWrap<TrackerType::ARANOVSKIY>>(dt);
    if (name == "zc")    return std::make_unique<WaveDirWrap<TrackerType::ZEROCROSS>>(dt);
    /* default */        return std::make_unique<WaveDirWrap<TrackerType::KALMANF>>(dt);
}

static void process_wave_file_direction_only(const std::string& filename,
                                             float dt,
                                             IWaveDir& dir)
{
    auto parsed = WaveFileNaming::parse_to_params(filename);
    if (!parsed) return;
    auto [kind, type, wp] = *parsed;
    if (kind != FileKind::Data) return;
    if (!(type == WaveType::JONSWAP || type == WaveType::PMSTOKES)) return;

    // Output filename: wdir_*.csv
    std::string outname = filename;
    auto pos_prefix = outname.find("wave_data_");
    if (pos_prefix != std::string::npos)
        outname.replace(pos_prefix, std::string("wave_data_").size(), "wdir_");
    else outname = "wdir_" + outname;

    auto pos_ext = outname.rfind(".csv");
    if (pos_ext == std::string::npos) outname += ".csv";

    std::cout << "Processing " << filename << " → " << outname << "\n";

    std::ofstream ofs(outname);
    ofs << "time,"
        << "disp_ref_x,disp_ref_y,disp_ref_z,"
        << "vel_ref_x,vel_ref_y,vel_ref_z,"
        << "acc_ref_x,acc_ref_y,acc_ref_z,"
        << "acc_meas_x,acc_meas_y,acc_meas_z,"
        << "freq_hz,phase,"
        << "dir_deg,dir_uncert_deg,dir_conf,dir_amp,"
        << "dir_vec_x,dir_vec_y,"
        << "dfilt_ax,dfilt_ay\n";

    // Noise models
    NoiseModel accel_noise = make_noise_model(0.03f, 0.02f, 1234);
    NoiseModel gyro_noise  = make_noise_model(0.001f, 0.0004f, 5678);
    (void)gyro_noise; // not used here, but kept for parity with your sim

    WaveDataCSVReader reader(filename);

    std::vector<float> times, freqs, dirs_deg, unc_deg, confs, amps;
    std::vector<int>   good_mask;          // 0/1 mask (no <cstdint> needed)
    constexpr float CONF_THRESH = 20.0f;   // same gates as your KalmanWaveDirection
    constexpr float AMP_THRESH  = 0.08f;

    reader.for_each_record([&](const Wave_Data_Sample &rec) {        
        // Raw BODY Z-up from CSV
        Vector3f acc_body(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        if (add_noise) acc_body = apply_noise(acc_body, accel_noise);
        
        // Update direction pipeline with BODY-frame accelerations
        dir.update(acc_body);

        // Reference (world Z-up) from file
        Vector3f disp_ref(rec.wave.disp_x, rec.wave.disp_y, rec.wave.disp_z);
        Vector3f vel_ref (rec.wave.vel_x,  rec.wave.vel_y,  rec.wave.vel_z);
        Vector3f acc_ref (rec.wave.acc_x,  rec.wave.acc_y,  rec.wave.acc_z);

        // Results
        const float f_hz        = dir.getFrequencyHz();
        const float phase       = dir.getPhase();
        const float dir_deg     = dir.getDirectionDegrees();
        const float dir_unc_deg = dir.getDirectionUncertaintyDegrees();
        const float dir_conf    = dir.getConfidence();
        const float dir_amp     = dir.getAmplitude();
        const Vector2f d        = dir.getDirectionUnit();
        const Vector2f dxy      = dir.getFilteredXY();

        times.push_back(rec.time);
        freqs.push_back(f_hz);
        dirs_deg.push_back(dir_deg);
        unc_deg.push_back(dir_unc_deg);
        confs.push_back(dir_conf);
        amps.push_back(dir_amp);
        good_mask.push_back((dir_conf > CONF_THRESH && dir_amp > AMP_THRESH) ? 1 : 0);
        
        // CSV row
        ofs << rec.time << ","
            << disp_ref.x() << "," << disp_ref.y() << "," << disp_ref.z() << ","
            << vel_ref.x()  << "," << vel_ref.y()  << "," << vel_ref.z()  << ","
            << acc_ref.x()  << "," << acc_ref.y()  << "," << acc_ref.z()  << ","
            << acc_body.x() << "," << acc_body.y() << "," << acc_body.z() << ","
            << f_hz << "," << phase << ","
            << dir_deg << "," << dir_unc_deg << "," << dir_conf << "," << dir_amp << ","
            << d.x() << "," << d.y() << ","
            << dxy.x() << "," << dxy.y() << "\n";
    });

    ofs.close();
    std::cout << "Wrote " << outname << "\n";

    // Post-run report (LAST 60 s ONLY)
    if (!times.empty()){
        const size_t N = times.size();
        const float  T_END = times.back();
        const float  WINDOW_S = 60.0f;
        const float  T0 = std::max(times.front(), T_END - WINDOW_S);

        // index of first sample >= T0
        size_t i0 = size_t(std::lower_bound(times.begin(), times.end(), T0) - times.begin());
        if (i0 > N) i0 = N;
        const size_t i1 = N;

        auto print_block = [&](const char* title, size_t i0, size_t i1){
            if (i0 >= i1) { std::cout << title << ": no data\n"; return; }

            std::vector<float> vf(freqs.begin()+i0,    freqs.begin()+i1);
            std::vector<float> vd(dirs_deg.begin()+i0, dirs_deg.begin()+i1);
            std::vector<float> vu(unc_deg.begin()+i0,  unc_deg.begin()+i1);
            std::vector<float> vc(confs.begin()+i0,    confs.begin()+i1);
            std::vector<float> va(amps.begin()+i0,     amps.begin()+i1);

            size_t good = 0; for (size_t k=i0; k<i1; ++k) good += good_mask[k];
            auto cs = circular_stats_180(vd);

            std::cout << title << "\n";
            std::cout << "  window_s: " << (times[i1-1] - times[i0]) << "\n";
            std::cout << "  samples: " << (i1 - i0) << "\n";
            std::cout << "  freq_hz: mean=" << mean(vf)
                      << "  median=" << median(vf)
                      << "  p05=" << percentile(vf,0.05)
                      << "  p95=" << percentile(vf,0.95) << "\n";
            std::cout << "  dir_deg (0..180): mean_circ=" << cs.mean_deg
                      << "  circ_std≈" << cs.std_deg << " deg\n";
            std::cout << "  uncert_deg: mean=" << mean(vu)
                      << "  median=" << median(vu)
                      << "  p95=" << percentile(vu,0.95) << "\n";
            std::cout << "  confidence: mean=" << mean(vc)
                      << "  >" << 20.0f << " count=" << good
                      << " (" << (100.0 * double(good)/double(i1-i0)) << "%)\n";
            std::cout << "  amplitude: mean=" << mean(va)
                      << "  median=" << median(va) << "\n";
        };

        std::cout << "=== Direction Report (last 60 s only) for " << outname << " ===\n";
        print_block("— Last 60 s —", i0, i1);
        std::cout << "=============================================\n\n";
    }
}

int main(int argc, char* argv[]) {
    float dt = 1.0f / 240.0f;

    std::string tracker_name = "kalmf"; // default
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-noise") add_noise = false;
        else if (arg.rfind("--tracker=", 0) == 0) {
            tracker_name = arg.substr(std::string("--tracker=").size());
        }
    }

    std::cout << "Wave direction run: tracker=" << tracker_name
              << ", noise=" << (add_noise ? "true" : "false")
              << ", dt=" << dt << " s\n";

    auto dir = make_dir_estimator(tracker_name, dt);

    std::vector<std::string> files;
    for (auto &entry : std::filesystem::directory_iterator(".")) {
        if (!entry.is_regular_file()) continue;
        const std::string fname = entry.path().string();
        if (fname.find("wave_data_") == std::string::npos) continue;
        if (auto kind = WaveFileNaming::parse_kind_only(fname);
            kind && *kind == FileKind::Data) {
            files.push_back(fname);
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto& fname : files)
        process_wave_file_direction_only(fname, dt, *dir);

    return 0;
}
