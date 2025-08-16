#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <limits>
#include <functional>
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
#include "SeaStateRegularity.h"

// --- Config ---
static constexpr float SAMPLE_RATE_HZ = 240.0f;
static constexpr float DELTA_T = 1.0f / SAMPLE_RATE_HZ;
static constexpr float TEST_DURATION_S = 5 * 60.0f;
static constexpr float WARMUP_SEC = 0.5f;   // warmup for 0.5s
static constexpr unsigned SEED_BASE = 239u;
static constexpr float NOISE_STDDEV = 0.08f;
static constexpr float BIAS_MEAN = 0.1f;

enum class WaveType { GERSTNER=0, JONSWAP=1, FENTON=2 };
enum class TrackerType { ARANOVSKIY=0, KALMANF=1, ZEROCROSS=2 };

struct WaveParameters {
    float freqHz;
    float height;
    float phase;
    float direction;
};

const std::vector<WaveParameters> waveParamsList = {
    {1.0f/3.0f,  0.135f, static_cast<float>(M_PI/3.0), 30.0f},
    {1.0f/5.7f,  0.75f,  static_cast<float>(M_PI/3.0), 30.0f},
    {1.0f/8.5f,  2.0f,   static_cast<float>(M_PI/3.0), 30.0f},
    {1.0f/11.4f, 4.25f,  static_cast<float>(M_PI/3.0), 30.0f},
    {1.0f/14.3f, 7.4f,   static_cast<float>(M_PI/3.0), 30.0f}
};

// Global trackers
AranovskiyFilter<double> arFilter;
KalmANF<double> kalmANF;
KalmanSmootherVars kalman_freq;
SchmittTriggerFrequencyDetector freqDetector(ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS);

// Make filename
static std::string make_filename(TrackerType tr, WaveType wt, float height) {
    std::string trackerName = (tr == TrackerType::ARANOVSKIY) ? "aranovskiy" :
                              (tr == TrackerType::KALMANF) ? "kalmANF" : "zerocrossing";
    std::string waveName = (wt == WaveType::GERSTNER) ? "gerstner" :
                           (wt == WaveType::JONSWAP) ? "jonswap" : "fenton";
    char buf[128];
    std::snprintf(buf, sizeof(buf), "regularity_%s_%s_h%.3f.csv",
                  trackerName.c_str(), waveName.c_str(), height);
    return std::string(buf);
}

// Wave sample struct
struct Wave_Sample { float accel_z; };

// Sample helpers
static Wave_Sample sample_gerstner(double t, TrochoidalWave<float> &wave_obj) {
    return { wave_obj.surfaceVerticalAcceleration(static_cast<float>(t)) };
}

template<int N=256>
static Wave_Sample sample_jonswap(double t, Jonswap3dGerstnerWaves<N> &model) {
    auto state = model.getLagrangianState(0.0f, 0.0f, static_cast<float>(t));
    return { state.acceleration.z() };
}

template<int ORD=4>
static Wave_Sample sample_fenton(double t, WaveSurfaceTracker<ORD> &tracker) {
    Wave_Sample s{};
    tracker.track_floating_object_step(static_cast<float>(t),
        [&](float time, float dt, float elevation, float velocity, float acceleration, float x, float vx){
            s.accel_z = acceleration;
        });
    return s;
}

// Run tracker once and return frequency
static double run_tracker_once(TrackerType tracker, float a_norm, float dt, uint32_t now_us) {
    double freq = std::numeric_limits<double>::quiet_NaN();
    if (tracker == TrackerType::ARANOVSKIY)
        freq = estimate_freq(Aranovskiy, &arFilter, &kalmANF, &freqDetector, a_norm, a_norm, dt, now_us);
    else if (tracker == TrackerType::KALMANF)
        freq = estimate_freq(Kalm_ANF, &arFilter, &kalmANF, &freqDetector, a_norm, a_norm, dt, now_us);
    else
        freq = estimate_freq(ZeroCrossing, &arFilter, &kalmANF, &freqDetector, a_norm, a_norm, dt, now_us);
    return freq;
}

// Process a single sample
static void process_sample(float noisy_accel, double sim_t, TrackerType tracker,
                           SeaStateRegularity& regFilter, std::ofstream& ofs) {
    float a_norm = noisy_accel / 9.81f;
    double freq = run_tracker_once(tracker, a_norm, DELTA_T, static_cast<uint32_t>(sim_t * 1e6));
    if (std::isfinite(freq)) {
        regFilter.update(DELTA_T, noisy_accel, static_cast<float>(2.0 * M_PI * freq));
        ofs << sim_t << ","
            << (2.0 * M_PI * freq) << ","
            << regFilter.getNarrowness() << ","
            << regFilter.getRegularity() << ","
            << regFilter.getWaveHeightEnvelopeEst() << ","
            << regFilter.getDisplacementFrequencyHz() << "\n";
    }
}

// Scenario runner
static void run_one_scenario(WaveType waveType, TrackerType tracker, const WaveParameters &wp, unsigned run_seed) {
    std::string filename = make_filename(tracker, waveType, wp.height);
    std::ofstream ofs(filename);
    if (!ofs.is_open()) { fprintf(stderr, "Failed to open %s\n", filename.c_str()); return; }
    ofs << "time,omega_inst,narrowness,regularity,significant_wave_height,disp_freq_hz\n";

    init_filters(&arFilter, &kalman_freq);
    init_filters_alt(&kalmANF, &kalman_freq);
    freqDetector.reset();

    std::default_random_engine rng(SEED_BASE + run_seed);
    std::normal_distribution<float> gauss(0.0f, NOISE_STDDEV);
    float bias = BIAS_MEAN;

    SeaStateRegularity regFilter;
    double sim_t = 0.0;
    int total_steps = static_cast<int>(std::ceil(TEST_DURATION_S * SAMPLE_RATE_HZ));
    int warmup_steps = static_cast<int>(std::ceil(WARMUP_SEC * SAMPLE_RATE_HZ));

    // Local wave objects
    TrochoidalWave<float> trocho(0,1,0);
    Jonswap3dGerstnerWaves<256> jonswap_model(0,1,0,0,0,0,9.81f,1.0f);
    FentonWave<4> fenton_wave(0,0,0,0);
    WaveSurfaceTracker<4> fenton_tracker(0,0,0,0,5.0f,0.1f);

    // Sample function
    std::function<Wave_Sample(double)> sample_func;
    if (waveType == WaveType::GERSTNER) {
        float period = 1.0f / wp.freqHz;
        trocho = TrochoidalWave<float>(wp.height, period, wp.phase);
        sample_func = [&](double t){ return sample_gerstner(t, trocho); };
    } else if (waveType == WaveType::JONSWAP) {
        float period = 1.0f / wp.freqHz;
        jonswap_model = Jonswap3dGerstnerWaves<256>(wp.height, period, wp.direction, 0.02f,0.8f,3.3f,9.81f,15.0f);
        sample_func = [&](double t){ return sample_jonswap(t, jonswap_model); };
    } else if (waveType == WaveType::FENTON) {
        auto fenton_params = FentonWave<4>::infer_fenton_parameters_from_amplitude(
            wp.height, 200.0f, 2.0f*M_PI*wp.freqHz, wp.phase);
        fenton_wave = FentonWave<4>(fenton_params.height, fenton_params.depth,
                                    fenton_params.length, fenton_params.initial_x);
        fenton_tracker = WaveSurfaceTracker<4>(fenton_params.height, fenton_params.depth,
                                               fenton_params.length, fenton_params.initial_x,
                                               5.0f,0.1f);
        sample_func = [&](double t){ return sample_fenton(t, fenton_tracker); };
    }

    // --- Warmup ---
    for (int i=0; i<warmup_steps; ++i) {
        auto samp = sample_func(sim_t);
        float a_norm = samp.accel_z / 9.81f;
        run_tracker_once(tracker, a_norm, DELTA_T, static_cast<uint32_t>(sim_t*1e6));
        sim_t += DELTA_T;
    }

    // --- Main loop ---
    if (waveType != WaveType::FENTON) {
        for (int i=0; i<total_steps; ++i) {
            auto samp = sample_func(sim_t);
            float noisy_accel = samp.accel_z + bias + gauss(rng);
            process_sample(noisy_accel, sim_t, tracker, regFilter, ofs);
            sim_t += DELTA_T;
        }
    } else {
        fenton_tracker.track_floating_object(TEST_DURATION_S, DELTA_T,
            [&](float time, float dt, float elevation, float velocity, float acceleration, float x, float vx){
                float noisy_accel = acceleration + bias + gauss(rng);
                process_sample(noisy_accel, sim_t, tracker, regFilter, ofs);
                sim_t = time;
            });
    }

    ofs.close();
    printf("Wrote %s\n", filename.c_str());
}

int main() {
    unsigned run_idx = 0;
    for (const auto& wp : waveParamsList) {
        for (int wt=0; wt<3; ++wt) {
            for (int tr=0; tr<3; ++tr) {
                run_one_scenario(static_cast<WaveType>(wt), static_cast<TrackerType>(tr), wp, run_idx++);
            }
        }
    }
    printf("All SeaStateRegularity runs complete.\n");
    return 0;
}
