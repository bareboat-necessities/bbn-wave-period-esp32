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

// ===== Eigen (Arduino/non-Arduino) =====
#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
#else
  #include <ArduinoEigenDense.h>
#endif

using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Matrix2f;

// ===== Project headers you already have =====
#include "WaveFilesSupport.h"
#include "FrameConversions.h"

#include "AranovskiyFilter.h"
#include "KalmANF.h"
#include "SchmittTriggerFrequencyDetector.h"
#include "FrequencySmoother.h"

// ===== Shared constants (match your sim) =====
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ZERO_CROSSINGS_SCALE          1.0f
#define ZERO_CROSSINGS_DEBOUNCE_TIME  0.12f
#define ZERO_CROSSINGS_STEEPNESS_TIME 0.21f

#define FREQ_GUESS 0.3f   // Hz

const float g_std = 9.80665f;     // standard gravity acceleration m/s²

// CLI & sim flags
static bool add_noise = true;

// ===== Noise model (same style you used) =====
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

// ===== KalmanWaveDirection (inlined from your header, unchanged except minor includes) =====
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

// ===== TrackerPolicy + runtime wrapper =====
enum class TrackerType { ARANOVSKIY, KALMANF, ZEROCROSS };

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

// WaveDirectionEstimator (templated) + runtime-virtual wrapper
constexpr float MIN_FREQ_HZ = 0.1f;
constexpr float MAX_FREQ_HZ = 5.0f;

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

// ===== Main processing (reads wave_data_*.csv; writes wdir_*.csv) =====
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

    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        // Raw BODY Z-up from CSV
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        if (add_noise) acc_b = apply_noise(acc_b, accel_noise);

        // Map BODY Z-up → BODY NED
        Vector3f acc_meas_ned = zu_to_ned(acc_b);

        // Update direction pipeline
        dir.update(acc_meas_ned);

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

        // CSV row
        ofs << rec.time << ","
            << disp_ref.x() << "," << disp_ref.y() << "," << disp_ref.z() << ","
            << vel_ref.x()  << "," << vel_ref.y()  << "," << vel_ref.z()  << ","
            << acc_ref.x()  << "," << acc_ref.y()  << "," << acc_ref.z()  << ","
            << acc_meas_ned.x() << "," << acc_meas_ned.y() << "," << acc_meas_ned.z() << ","
            << f_hz << "," << phase << ","
            << dir_deg << "," << dir_unc_deg << "," << dir_conf << "," << dir_amp << ","
            << d.x() << "," << d.y() << ","
            << dxy.x() << "," << dxy.y() << "\n";
    });

    ofs.close();
    std::cout << "Wrote " << outname << "\n";
}

// ===== Main =====
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
