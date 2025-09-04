#include "WaveFilesSupport.h"
#include "TrochoidalWave.h"
#include "Jonswap3dStokesWaves.h"
#include "FentonWaveVectorized.h"
#include "PiersonMoskowitzStokes3D_Waves.h"
#include "DirectionalSpread.h"
#include "CnoidalWave.h"

// === Experiment Config ===
static constexpr float SAMPLE_RATE_HZ  = 240.0f;
static constexpr float DELTA_T         = 1.0f / SAMPLE_RATE_HZ;
static constexpr float TEST_DURATION_S = 10 * 60.0f;    // 10 minutes

// Example test cases
const std::vector<WaveParameters> waveParamsList = {
    {3.0f,   0.135f, static_cast<float>(M_PI/3.0),   30.0f},
    {5.7f,   0.75f,  static_cast<float>(M_PI/3.0),  -45.0f},
    {8.5f,   2.0f,   static_cast<float>(-M_PI/6.0),  60.0f},
    {11.4f,  4.25f,  static_cast<float>(M_PI/2.0), -120.0f},
    {14.3f,  7.4f,   static_cast<float>(-M_PI/2.0),  90.0f}
};

// === Sampling Helpers ===
static Wave_Data_Sample sample_gerstner(double t, TrochoidalWave<float> &wave_obj) {
    Wave_Data_Sample out{};
    out.time        = t;
    out.wave.disp_z = wave_obj.surfaceElevation(static_cast<float>(t));
    out.wave.vel_z  = wave_obj.surfaceVerticalVelocity(static_cast<float>(t));
    out.wave.acc_z  = wave_obj.surfaceVerticalAcceleration(static_cast<float>(t));
    return out;
}

template<int N=128>
static Wave_Data_Sample sample_jonswap(double t, Jonswap3dStokesWaves<N> &model) {
    auto state = model.getLagrangianState(0.0, 0.0, t, 0.0);
    Wave_Data_Sample out{};
    out.time        = t;
    out.wave.disp_z = state.position.z();
    out.wave.vel_z  = state.velocity.z();
    out.wave.acc_z  = state.acceleration.z();
    return out;
}

template<int N=128, int ORDER=3>
static Wave_Data_Sample sample_pmstokes(double t, PMStokesN3dWaves<N, ORDER> &model) {
    auto state = model.getLagrangianState(t);
    Wave_Data_Sample out{};
    out.time        = t;
    out.wave.disp_z = state.position.z();
    out.wave.vel_z  = state.velocity.z();
    out.wave.acc_z  = state.acceleration.z();
    return out;
}

template<int ORDER=4>
static std::vector<Wave_Data_Sample> sample_fenton(
        const WaveParameters &wp,
        double duration,
        double dt) 
{
    std::vector<Wave_Data_Sample> results;

    auto fenton_params = FentonWave<ORDER>::infer_fenton_parameters_from_amplitude(
        wp.height, 200.0f, 2.0f * M_PI / wp.period, wp.phase);

    WaveSurfaceTracker<ORDER> fenton_tracker(
        fenton_params.height,
        fenton_params.depth,
        fenton_params.length,
        fenton_params.initial_x,
        5.0f,    // mass (kg)
        0.1f     // drag coeff
    );

    auto callback = [&](float time, float dt, float elevation,
                        float vertical_velocity, float vertical_acceleration,
                        float x, float vx) {
        (void)dt; (void)x; (void)vx;
        Wave_Data_Sample out{};
        out.time        = time;
        out.wave.disp_z = elevation;
        out.wave.vel_z  = vertical_velocity;
        out.wave.acc_z  = vertical_acceleration;
        results.push_back(out);
    };

    fenton_tracker.track_floating_object(duration, dt, callback);
    return results;
}

static Wave_Data_Sample sample_cnoidal(double t, CnoidalWave<float> &wave) {
    Wave_Data_Sample out{};
    out.time        = t;
    out.wave.disp_z = wave.surfaceElevation(0.0f, 0.0f, t);
    out.wave.vel_z  = wave.wVelocity(0.0f, 0.0f, 0.0f, t);
    out.wave.acc_z  = wave.azAcceleration(0.0f, 0.0f, 0.0f, t);
    return out;
}

// === Scenario Runner ===
static void run_one_scenario(WaveType waveType, const WaveParameters &wp) {
    WaveParameters wp_copy = wp;
    // Zero direction for models that ignore it
    if (waveType == WaveType::GERSTNER || 
        waveType == WaveType::FENTON   || 
        waveType == WaveType::CNOIDAL) {
        wp_copy.direction = 0.0f;
    }

    std::string filename = WaveFileNaming::generate(waveType, wp_copy);

    WaveDataCSVWriter writer(filename);
    writer.write_header();

    double sim_t = 0.0;
    int total_steps = static_cast<int>(std::ceil(TEST_DURATION_S * SAMPLE_RATE_HZ));

    if (waveType == WaveType::GERSTNER) {
        TrochoidalWave<float> trocho(wp.height, wp.period, wp.phase);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_gerstner(sim_t, trocho);
            writer.write(samp);
            sim_t += DELTA_T;
        }
    }
    else if (waveType == WaveType::JONSWAP) {
        auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
            wp.direction * M_PI / 180.0, 10.0, GLOBAL_SEED);
        auto jonswap_model = std::make_unique<Jonswap3dStokesWaves<128>>(
            wp.height, wp.period, dirDist, 0.02, 0.8, 3.3, g_std, GLOBAL_SEED);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_jonswap(sim_t, *jonswap_model);
            writer.write(samp);
            sim_t += DELTA_T;
        }
    }
    else if (waveType == WaveType::FENTON) {
        auto samples = sample_fenton<4>(wp, TEST_DURATION_S, DELTA_T);
        for (auto &samp : samples) {
            writer.write(samp);
        }
    }
    else if (waveType == WaveType::PMSTOKES) {
        auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
            wp.direction * M_PI / 180.0, 10.0, GLOBAL_SEED);
        PMStokesN3dWaves<128, 3> waveModel(
            wp.height, wp.period, dirDist, 0.02, 0.8, g_std, GLOBAL_SEED);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_pmstokes(sim_t, waveModel);
            writer.write(samp);
            sim_t += DELTA_T;
        }
    }
    else if (waveType == WaveType::CNOIDAL) {
        CnoidalWave<float> cnoidal(200.0f, wp.height, wp.period, 0.0f, g_std);
        for (int step = 0; step < total_steps; ++step) {
            auto samp = sample_cnoidal(sim_t, cnoidal);
            writer.write(samp);
            sim_t += DELTA_T;
        }
    }

    writer.close();
    printf("Wrote %s\n", filename.c_str());
}

// === Main ===
int main() {
    for (const auto& wp : waveParamsList) {
        for (int wt = 0; wt <= static_cast<int>(WaveType::CNOIDAL); ++wt) {
            run_one_scenario(static_cast<WaveType>(wt), wp);
        }
    }
    printf("All wave data generation complete.\n");
    return 0;
}
