#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <limits>
#include <iomanip>

#include "AranovskiyFilter.h"
#include "KalmANF.h"
#include "FrequencySmoother.h"
#include "KalmanForWaveBasic.h"
#include "KalmanWaveNumStableAlt.h"
#include "SchmittTriggerFrequencyDetector.h"
#include "KalmanWaveDirection.h"
#include "TrochoidalWave.h"
#include "KalmanSmoother.h"
#include "WaveFilters.h"
#include "Jonswap3D_Waves.h"
#include "FentonWaveVectorized.h"
#include "WaveSurfaceProfile.h"

// Configuration
static constexpr float SAMPLE_RATE_HZ = 240.0f;
static constexpr float DELTA_T = 1.0f / SAMPLE_RATE_HZ;
static constexpr float TEST_DURATION_S = 5 * 60.0f;   // seconds per run
static constexpr unsigned SEED_BASE = 239u;
static constexpr float NOISE_STDDEV = 0.08f;
static constexpr float BIAS_MEAN = 0.1f;      // m/s^2 bias added to accel

enum class WaveType { GERSTNER=0, JONSWAP=1, FENTON=2 };
enum class TrackerType { ARANOVSKIY=0, KALMANF=1, ZEROCROSS=2 };

static std::string make_filename(TrackerType tr, WaveType wt, float height) {
    std::string trackerName = (tr == TrackerType::ARANOVSKIY) ? "aranovskiy" :
                              (tr == TrackerType::KALMANF) ? "kalmANF" : "zerocrossing";
    std::string waveName = (wt == WaveType::GERSTNER) ? "gerstner" :
                           (wt == WaveType::JONSWAP) ? "jonswap" : "fenton";
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s_%s_h%.3f.csv", trackerName.c_str(), waveName.c_str(), height);
    return std::string(buf);
}

struct WaveParameters {
    float freqHz;    // true frequency (Hz)
    float height;    // amplitude (m)
    float phase;     // radians
    float direction; // degrees (used for JONSWAP)
};

const std::vector<WaveParameters> waveParamsList = {
    {1.0f/3.0f,  0.135f, static_cast<float>(M_PI/3.0), 30.0f},
    {1.0f/5.7f,  0.75f,  static_cast<float>(M_PI/3.0), 30.0f},
    {1.0f/8.5f,  2.0f,   static_cast<float>(M_PI/3.0), 30.0f},
    {1.0f/11.4f, 4.25f,  static_cast<float>(M_PI/3.0), 30.0f},
    {1.0f/14.3f, 7.4f,   static_cast<float>(M_PI/3.0), 30.0f}
};

AranovskiyFilter<double> arFilter;
KalmANF<double> kalmANF;
FrequencySmoother<float> freqSmoother;
KalmanSmootherVars kalman_freq; // used for smoothing outputs (per-run reset)
SchmittTriggerFrequencyDetector freqDetector(ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS);

// current sim time (per run)
static double sim_t = 0.0;

// get time in microseconds
static uint32_t now_us() { return static_cast<uint32_t>(sim_t * 1e6); }

// Initialize trackers
static void init_tracker_backends() {
    init_filters(&arFilter, &kalman_freq);
    init_filters_alt(&kalmANF, &kalman_freq);
}

static bool kalm_smoother_first;

static void reset_run_state() {
    sim_t = 0.0;
    kalm_smoother_first = true;
    kalman_smoother_init(&kalman_freq, 0.25f, 2.0f, 100.0f);
    freqSmoother = FrequencySmoother<float>();
}

static double clamp_freq(double v) {
    return clamp(v, (double)FREQ_LOWER, (double)FREQ_UPPER);
}

static void write_csv_header(std::ofstream &ofs) {
    ofs << "time,true_freq,est_freq,smooth_freq,error,smooth_error,abs_error,abs_smooth_error,update_flag\n";
}

static void write_csv_line(std::ofstream &ofs,
                           double time, double true_freq,
                           double est_freq, double smooth_freq,
                           double error, double smooth_error,
                           double abs_error, double abs_smooth_error,
                           bool updated) {
    ofs << std::fixed << std::setprecision(6)
        << time << ","
        << true_freq << ",";
    if (std::isnan(est_freq)) ofs << "nan,";
    else ofs << est_freq << ",";
    if (std::isnan(smooth_freq)) ofs << "nan,";
    else ofs << smooth_freq << ",";
    if (std::isnan(error)) ofs << "nan,";
    else ofs << error << ",";
    if (std::isnan(smooth_error)) ofs << "nan,";
    else ofs << smooth_error << ",";
    if (std::isnan(abs_error)) ofs << "nan,";
    else ofs << abs_error << ",";
    if (std::isnan(abs_smooth_error)) ofs << "nan,";
    else ofs << abs_smooth_error << ",";
    ofs << (updated ? 1 : 0) << "\n";
}

// Wave_Sample struct
struct Wave_Sample {
    float accel_z;
    float vel_z;
    float elev;
};

// Sample functions
static Wave_Sample sample_gerstner(const WaveParameters &p, double t, TrochoidalWave<float> &wave_obj) {
    Wave_Sample s;
    (void)p;
    s.accel_z = wave_obj.surfaceVerticalAcceleration(static_cast<float>(t));
    s.vel_z = wave_obj.surfaceVerticalSpeed(static_cast<float>(t));
    s.elev = wave_obj.surfaceElevation(static_cast<float>(t));
    return s;
}

template<int N=256>
static Wave_Sample sample_jonswap(const WaveParameters &p, double t, Jonswap3dGerstnerWaves<N> &model) {
    Wave_Sample s;
    auto state = model.getLagrangianState(0.0f, 0.0f, static_cast<float>(t));
    s.accel_z = state.acceleration.z();
    s.vel_z = state.velocity.z();
    s.elev = state.displacement.z();
    return s;
}

template<int ORD=4>
static Wave_Sample sample_fenton(const WaveParameters &p, double t, FentonWave<ORD> &fenton) {
    Wave_Sample s;
    s.accel_z = fenton.surfaceVerticalAcceleration(static_cast<float>(t));
    s.vel_z = fenton.surfaceVerticalSpeed(static_cast<float>(t));
    s.elev = fenton.surfaceElevation(static_cast<float>(t));
    return s;
}

static std::pair<double,bool> run_tracker_once(TrackerType tracker, float a_norm, float a_raw, float dt) {
    double freq = FREQ_GUESS;
    if (tracker == TrackerType::ARANOVSKIY) {
        freq = estimate_freq(Aranovskiy, &arFilter, &kalmANF, &freqDetector, a_norm, a_norm, dt, now_us());
    } else if (tracker == TrackerType::KALMANF) {
        freq = estimate_freq(Kalm_ANF, &arFilter, &kalmANF, &freqDetector, a_norm, a_norm, dt, now_us());
    } else {
        freq = estimate_freq(ZeroCrossing, &arFilter, &kalmANF, &freqDetector, a_norm, a_norm, dt, now_us());
    }
    bool updated = !std::isnan(freq);
    return { freq, updated };
}

static void run_one_scenario(WaveType waveType, TrackerType tracker, const WaveParameters &wp, unsigned run_seed) {
    std::string filename = make_filename(tracker, waveType, wp.height);
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        fprintf(stderr, "Failed to open %s\n", filename.c_str());
        return;
    }
    write_csv_header(ofs);
    reset_run_state();

    std::default_random_engine rng(SEED_BASE + run_seed);
    std::normal_distribution<float> gauss(0.0f, NOISE_STDDEV);
    float bias = BIAS_MEAN;

    kalman_smoother_init(&kalman_freq, 0.25f, 2.0f, 100.0f);
    kalm_smoother_first = true;

    // Common per-sample processing lambda
    auto process_sample = [&](float noisy_accel, float dt, double current_time) {
        float a_norm = noisy_accel / g_std;
        auto [est_freq, updated] = run_tracker_once(tracker, a_norm, noisy_accel, dt);
        double true_f = wp.freqHz;
        float smooth_freq = std::numeric_limits<float>::quiet_NaN();
        if (!std::isnan(est_freq) && updated) {
            if (kalm_smoother_first) {
                kalm_smoother_first = false;
                freqSmoother.setInitial(est_freq);
                smooth_freq = est_freq;
            } else {
                smooth_freq = freqSmoother.update(est_freq);
            }
        }
        if (!std::isnan(smooth_freq))
            smooth_freq = clamp_freq(smooth_freq);
        double error = std::isnan(est_freq) ? std::numeric_limits<double>::quiet_NaN() : (est_freq - true_f);
        double smooth_error = std::isnan(smooth_freq) ? std::numeric_limits<double>::quiet_NaN() : (smooth_freq - true_f);
        double abs_error = std::isnan(error) ? std::numeric_limits<double>::quiet_NaN() : std::fabs(error);
        double abs_smooth_error = std::isnan(smooth_error) ? std::numeric_limits<double>::quiet_NaN() : std::fabs(smooth_error);
        write_csv_line(ofs, current_time, true_f, est_freq, smooth_freq, error, smooth_error, abs_error, abs_smooth_error, updated);
    };

    if (waveType == WaveType::GERSTNER) {
        float period = 1.0f / wp.freqHz;
        TrochoidalWave<float> trocho(wp.height, period, wp.phase);
        int total_steps = static_cast<int>(std::ceil(TEST_DURATION_S * SAMPLE_RATE_HZ));
        for (int step = 0; step < total_steps; ++step) {
            Wave_Sample samp = sample_gerstner(wp, sim_t, trocho);
            float noisy_accel = samp.accel_z + bias + gauss(rng);
            process_sample(noisy_accel, DELTA_T, sim_t);
            sim_t += DELTA_T;
        }
    } else if (waveType == WaveType::JONSWAP) {
        float period = 1.0f / wp.freqHz;
        Jonswap3dGerstnerWaves<256> jonswap_model(wp.height, period, wp.direction, 0.02f, 0.8f, 3.3f, g_std, 15.0f);
        int total_steps = static_cast<int>(std::ceil(TEST_DURATION_S * SAMPLE_RATE_HZ));
        for (int step = 0; step < total_steps; ++step) {
            Wave_Sample samp = sample_jonswap(wp, sim_t, jonswap_model);
            float noisy_accel = samp.accel_z + bias + gauss(rng);
            process_sample(noisy_accel, DELTA_T, sim_t);
            sim_t += DELTA_T;
        }
    } else if (waveType == WaveType::FENTON) {
        auto fenton_params = FentonWave<4>::infer_fenton_parameters_from_amplitude(wp.height, 200.0f, 2.0f * M_PI * wp.freqHz, wp.phase);
        FentonWave<4> fenton_wave(fenton_params.height, fenton_params.depth, fenton_params.length, fenton_params.initial_x);
        WaveSurfaceTracker<4> fenton_tracker(
            fenton_params.height,
            fenton_params.depth,
            fenton_params.length,
            fenton_params.initial_x,
            5.0f,    // mass (kg)
            0.1f     // drag coeff
        );
        auto callback = [&](float time, float dt, float elevation, float vertical_velocity, float vertical_acceleration, float x, float vx) {
            float noisy_accel = vertical_acceleration + bias + gauss(rng);
            process_sample(noisy_accel, dt, time);

            sim_t = time; // keep sim_t updated
        };
        fenton_tracker.track_floating_object(TEST_DURATION_S, DELTA_T, callback);
    }
    ofs.close();
    printf("Wrote %s\n", filename.c_str());
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    init_tracker_backends();
    
    unsigned run_idx = 0;
    for (const auto& wp : waveParamsList) {
        for (int wt = 0; wt < 3; ++wt) {
            for (int tr = 0; tr < 3; ++tr) {
                reset_run_state();
                init_filters(&arFilter, &kalman_freq);
                init_filters_alt(&kalmANF, &kalman_freq);
                freqDetector.reset();

                run_one_scenario(static_cast<WaveType>(wt), static_cast<TrackerType>(tr), wp, run_idx);
                run_idx++;
            }
        }
    }
    printf("All runs complete.\n");
    return 0;
}
