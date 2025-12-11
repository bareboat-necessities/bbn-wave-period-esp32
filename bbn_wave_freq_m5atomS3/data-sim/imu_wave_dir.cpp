/*
  Wave direction runner using SeaStateFusionFilter
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Standard gravity (NED, +Z down)
const float g_std = 9.80665f;

#ifndef FREQ_GUESS
#define FREQ_GUESS 0.3f   // Hz (for initial ω inside filters that use it)
#endif

#define ZERO_CROSSINGS_SCALE          1.0f
#define ZERO_CROSSINGS_DEBOUNCE_TIME  0.12f
#define ZERO_CROSSINGS_STEEPNESS_TIME 0.21f

#include "WaveFilesSupport.h"
#include "FrameConversions.h"
#include "SeaStateFusionFilter.h"   // ← uses internal tracker (KalmANF/Aranovskiy/ZC) + dir filter

// CLI & sim flags
static bool add_noise = true;

struct NoiseModel {
    std::mt19937 rng;
    std::normal_distribution<float> dist;
    Vector3f bias;
};
static NoiseModel make_noise_model(float sigma, float bias_range, unsigned seed) {
    NoiseModel m{std::mt19937(seed),
                 std::normal_distribution<float>(0.0f, sigma),
                 Vector3f::Zero()};
    std::uniform_real_distribution<float> ub(-bias_range, bias_range);
    m.bias = Vector3f(ub(m.rng), ub(m.rng), ub(m.rng));
    return m;
}
static Vector3f apply_noise(const Vector3f& v, NoiseModel& m) {
    return v - m.bias + Vector3f(m.dist(m.rng), m.dist(m.rng), m.dist(m.rng));
}

// Stats helpers
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
static T percentile(std::vector<T> v, double p01){
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

static inline const char* wave_dir_to_cstr(WaveDirection w){
    switch (w){
        case FORWARD:     return "TOWARD";
        case BACKWARD:    return "AWAY";
        default:          return "UNCERTAIN";
    }
}
static inline int wave_dir_to_num(WaveDirection w){
    switch (w){
        case FORWARD:     return +1;
        case BACKWARD:    return -1;
        default:          return 0;
    }
}

// Circular mean/std on [0,180)
struct CircStats {
    float mean_deg = NAN;
    float std_deg  = NAN;
};
static CircStats circular_stats_180(const std::vector<float>& degs){
    CircStats cs;
    if (degs.empty()) return cs;
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
    double a_mean  = 0.5 * a2_mean;
    float md = float(rad_to_deg(a_mean));
    if (md < 0) md += 180.0f;
    cs.mean_deg = md;
    cs.std_deg = (R > 1e-9) ? float(rad_to_deg(0.5 * std::sqrt(std::max(0.0, -2.0*std::log(R))))) : 90.0f;
    return cs;
}

// Runtime wrapper over SeaStateFusionFilter<TrackerType>
struct IFusion {
    virtual ~IFusion() = default;
    virtual void  update(float dt, const Vector3f& gyro_body_ned, const Vector3f& acc_body_ned, float tempC=35.0f) = 0;

    // telemetry
    virtual float     freq_hz()   const = 0;
    virtual float     phase()     const = 0;
    virtual float     dir_deg()   const = 0;
    virtual float     dir_unc()   const = 0;
    virtual float     dir_conf()  const = 0;
    virtual float     dir_amp()   const = 0;
    virtual Vector2f  dir_vec()   const = 0;
    virtual Vector2f  dfilt_xy()  const = 0;
    virtual WaveDirection dir_sign_state() const = 0;
};

template<TrackerType T>
struct FusionWrap : IFusion {
    SeaStateFusionFilter<T> f;

    FusionWrap() {
        // initialize MEKF noise std devs (tweak as needed)
        const Vector3f sigma_a(0.03f,  0.03f,  0.03f);
        const Vector3f sigma_g(0.001f, 0.001f, 0.001f);
        const Vector3f sigma_m(0.02f,  0.02f,  0.02f);
        f.initialize(sigma_a, sigma_g, sigma_m);

        // Optional: tune dir filter noises (BODY XY)
        f.dir().setMeasurementNoise(0.03f * 0.03f);
        f.dir().setProcessNoise(1e-6f);
    }

    void update(float dt, const Vector3f& gyro_body_ned, const Vector3f& acc_body_ned, float tempC=35.0f) override {
        f.updateTime(dt, gyro_body_ned, acc_body_ned, tempC);
    }

    // Telemetry passthrough
    float    freq_hz()   const override { return f.getFreqHz(); }
    float    phase()     const override { return f.dir().getPhase(); }
    float    dir_deg()   const override { return f.dir().getDirectionDegrees(); }             // [0,180)
    float    dir_unc()   const override { return f.dir().getDirectionUncertaintyDegrees(); }
    float    dir_conf()  const override { return f.dir().getLastStableConfidence(); }
    float    dir_amp()   const override { return f.dir().getAmplitude(); }
    Vector2f dir_vec()   const override { return f.dir().getDirection(); }
    Vector2f dfilt_xy()  const override { return f.dir().getFilteredSignal(); }
    WaveDirection dir_sign_state() const override { return f.getDirSignState(); }
};

static std::unique_ptr<IFusion> make_fusion(const std::string& name, bool with_mag=false) {
    (void)with_mag; // currently unused; SeaStateFusionFilter ctor has with_mag arg if you want it
    if (name == "aran") return std::make_unique<FusionWrap<TrackerType::ARANOVSKIY>>();
    if (name == "zc")   return std::make_unique<FusionWrap<TrackerType::ZEROCROSS>>();
    return std::make_unique<FusionWrap<TrackerType::KALMANF>>(); // default
}

// Processing
static void process_wave_file_direction_only(const std::string& filename,
                                             float dt, IFusion& fusion)
{
    auto parsed = WaveFileNaming::parse_to_params(filename);
    if (!parsed) return;
    auto [kind, type, wp] = *parsed;
    if (kind != FileKind::Data) return;
    if (!(type == WaveType::JONSWAP || type == WaveType::PMSTOKES)) return;

    // Output filename: wdir_*.csv
    std::string outname = filename;
    const auto pos_prefix = outname.find("wave_data_");
    if (pos_prefix != std::string::npos)
        outname.replace(pos_prefix, std::string("wave_data_").size(), "wdir_");
    else outname = "wdir_" + outname;

    const auto pos_ext = outname.rfind(".csv");
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
        << "dir_sign,dir_sign_num," 
        << "dir_vec_x,dir_vec_y,"
        << "dfilt_ax,dfilt_ay\n";

    // Noise models
    NoiseModel accel_noise = make_noise_model(0.03f, 0.02f, 1234);
    NoiseModel gyro_noise  = make_noise_model(0.001f, 0.0004f, 5678);

    WaveDataCSVReader reader(filename);

    std::vector<float> times, freqs, dirs_deg, unc_deg, confs, amps;
    std::vector<int>   good_mask;
    std::vector<int>   signs_num;  // -1 AWAY, 0 UNCERTAIN, +1 TOWARD
  
    constexpr float CONF_THRESH = 20.0f;   // same gates as KalmanWaveDirection
    constexpr float AMP_THRESH  = 0.08f;

    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        // BODY-frame IMU from CSV (Z-up in file spec; runner uses NED BODY)
        Vector3f acc_body(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3f gyro_body(0.0f, 0.0f, 0.0f);
        // If you have gyro in CSV (e.g., rec.imu.gyro_bx/by/bz), map it here:
        // gyro_body = { rec.imu.gyro_bx, rec.imu.gyro_by, rec.imu.gyro_bz };

        if (add_noise) {
            acc_body  = apply_noise(acc_body,  accel_noise);
            gyro_body = apply_noise(gyro_body, gyro_noise);
        }

        // Run fusion (MEKF + tracker + smoother + direction)
        fusion.update(dt, gyro_body, acc_body, /*tempC=*/35.0f);

        // Reference WORLD (Z-up) from file
        Vector3f disp_ref(rec.wave.disp_x, rec.wave.disp_y, rec.wave.disp_z);
        Vector3f vel_ref (rec.wave.vel_x,  rec.wave.vel_y,  rec.wave.vel_z);
        Vector3f acc_ref (rec.wave.acc_x,  rec.wave.acc_y,  rec.wave.acc_z);

        // Results
        const float f_hz         = fusion.freq_hz();
        const float phase        = fusion.phase();
        const float dir_deg      = fusion.dir_deg();
        const float dir_unc_deg  = fusion.dir_unc();
        const float dir_conf     = fusion.dir_conf();
        const float dir_amp      = fusion.dir_amp();
        const Vector2f d         = fusion.dir_vec();
        const Vector2f dxy       = fusion.dfilt_xy();
        const WaveDirection sign = fusion.dir_sign_state();
        const char* sign_str     = wave_dir_to_cstr(sign);
        const int   sign_num     = wave_dir_to_num(sign);
      
        times.push_back(rec.time);
        freqs.push_back(f_hz);
        dirs_deg.push_back(dir_deg);
        unc_deg.push_back(dir_unc_deg);
        confs.push_back(dir_conf);
        amps.push_back(dir_amp);
        signs_num.push_back(sign_num);
        good_mask.push_back((dir_conf > CONF_THRESH && dir_amp > AMP_THRESH) ? 1 : 0);

        ofs << rec.time << ","
            << disp_ref.x() << "," << disp_ref.y() << "," << disp_ref.z() << ","
            << vel_ref.x()  << "," << vel_ref.y()  << "," << vel_ref.z()  << ","
            << acc_ref.x()  << "," << acc_ref.y()  << "," << acc_ref.z()  << ","
            << acc_body.x() << "," << acc_body.y() << "," << acc_body.z() << ","
            << f_hz << "," << phase << ","
            << dir_deg << "," << dir_unc_deg << "," << dir_conf << "," << dir_amp << ","
            << sign_str << "," << sign_num << "," 
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

            int nToward = 0, nAway = 0, nUnc = 0;
            for (size_t k=i0; k<i1; ++k){
                const int s = signs_num[k];
                if (s > 0) ++nToward; else if (s < 0) ++nAway; else ++nUnc;
            }
            const int nWin = int(i1 - i0);
            auto pct = [&](int n){ return (nWin > 0) ? (100.0 * double(n) / double(nWin)) : 0.0; };
          
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
            std::cout << "  sign: TOWARD=" << nToward << " (" << pct(nToward) << "%)"
                      << "  AWAY=" << nAway << " (" << pct(nAway) << "%)"
                      << "  UNCERTAIN=" << nUnc << " (" << pct(nUnc) << "%)\n";
        };

        std::cout << "=== Direction Report (last 60 s only) for " << outname << " ===\n";
        print_block("— Last 60 s —", i0, i1);
        std::cout << "=============================================\n\n";
    }
}

int main(int argc, char* argv[]) {
    const float dt = 1.0f / 240.0f;

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

    auto fusion = make_fusion(tracker_name, /*with_mag*/ true);

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
        process_wave_file_direction_only(fname, dt, *fusion);

    return 0;
}

