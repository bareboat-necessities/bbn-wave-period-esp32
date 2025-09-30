#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <random>

#define EIGEN_NON_ARDUINO
#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float g_std = 9.80665f;     // standard gravity acceleration m/s²

#include "WaveFilesSupport.h"      // WaveDataCSVReader, Wave_Data_Sample, WaveParameters
#include "FrameConversions.h"      // zu_to_ned
#include "WaveSpectrumEstimator.h" // estimator
#include "Jonswap3dStokesWaves.h"  // reference spectrum
#include "PiersonMoskowitzStokes3D_Waves.h"

using Eigen::Vector3f;

// Noise model
struct NoiseModel {
    std::default_random_engine rng;
    std::normal_distribution<float> dist;
    Vector3f bias;
};

NoiseModel make_noise_model(float sigma, float bias_range, unsigned seed) {
    NoiseModel m{std::default_random_engine(seed),
                 std::normal_distribution<float>(0.0f, sigma),
                 Vector3f::Zero()};
    std::uniform_real_distribution<float> ub(-bias_range, bias_range);
    m.bias = Vector3f(ub(m.rng), ub(m.rng), ub(m.rng));
    return m;
}

Vector3f apply_noise(const Vector3f &v, NoiseModel &m) {
    return v + m.bias + Vector3f(m.dist(m.rng), m.dist(m.rng), m.dist(m.rng));
}

// ============================
// Spectrum simulation
// ============================
void process_wave_file(const std::string &filename, float dt) {
    auto parsed = WaveFileNaming::parse_to_params(filename);
    if (!parsed) return;
    auto [kind, type, wp] = *parsed;
    if (kind != FileKind::Data) return;

    std::cout << "Processing " << filename
              << " (" << EnumTraits<WaveType>::to_string(type)
              << "), Hs=" << wp.height << " m, Tp=" << wp.period << " s\n";

    // Configure noise model (same as Kalman sim)
    static NoiseModel accel_noise = make_noise_model(0.03f, 0.02f, 1234);

    // Estimator
    constexpr int Nfreq = 32;
    constexpr int Nblock = 2048;
    WaveSpectrumEstimator<Nfreq, Nblock> estimator(240.0, 50, true);

    // Reference model spectrum (higher-res, then interpolate to estimator freqs)
    Eigen::Matrix<double,128,1> f_ref, S_ref;
    if (type == WaveType::JONSWAP) {
        auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
        wp.direction * M_PI / 180.0, 10.0, 42u);

        Jonswap3dStokesWaves<128> refModel(wp.height, wp.period, dirDist,
                                           0.03, 1.5, 3.3, g_std, 42u);
        f_ref = refModel.frequencies();
        S_ref = refModel.spectrum();
    } else if (type == WaveType::PMSTOKES) {
        auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
        wp.direction * M_PI / 180.0, 10.0, 42u);

        PMStokesN3dWaves<128,3> refModel(wp.height, wp.period, dirDist,
                                         0.03, 1.5, g_std, 42u);
        f_ref = refModel.frequencies();
        S_ref = refModel.spectrum();
    }

    // Build CSV outname: wave_data_ → spectrum_
    std::string outname = filename;
    auto pos_prefix = outname.find("wave_data_");
    if (pos_prefix != std::string::npos) {
        outname.replace(pos_prefix, std::string("wave_data_").size(), "spectrum_");
    } else {
        outname = "spectrum_" + outname;
    }

    std::ofstream ofs(outname);
    ofs << "block,time,Hs,Fp,PM_alpha,PM_fp,PM_cost";
    auto freqs = estimator.getFrequencies();
    for (int i = 0; i < Nfreq; i++) {
        ofs << ",S_eta_est_f" << i << "_Hz=" << freqs[i]
            << ",S_eta_ref_f" << i;
    }
    ofs << "\n";

    WaveDataCSVReader reader(filename);
    int block_count = 0;

    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        // True acceleration in body frame
        Vector3f acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);

        // Apply noise
        Vector3f acc_noisy = apply_noise(acc_b, accel_noise);

        // Transform to world (NED)
        Vector3f acc_f = zu_to_ned(acc_noisy);

        // Vertical acceleration = Down component
        double a_vert = static_cast<double>(acc_f.z());

        // Feed into estimator
        if (estimator.processSample(a_vert)) {
            block_count++;

            auto S_est = estimator.getDisplacementSpectrum();
            double Hs = estimator.computeHs();
            double Fp = estimator.estimateFp();
            auto pm = estimator.fitPiersonMoskowitz();

            std::cout << "  Block " << block_count
                      << ": Hs=" << Hs << " m"
                      << ", Fp=" << Fp << " Hz"
                      << ", PM(fp=" << pm.fp
                      << ", alpha=" << pm.alpha
                      << ", cost=" << pm.cost << ")\n";

            // Interpolate reference spectrum onto estimator freqs
            std::vector<double> S_ref_interp(Nfreq, 0.0);
            for (int i = 0; i < Nfreq; i++) {
                double f = freqs[i];
                // linear search (small N=128, fine)
                int j = 0;
                while (j < f_ref.size()-1 && f_ref(j+1) < f) j++;
                if (j < f_ref.size()-1) {
                    double t = (f - f_ref(j)) / (f_ref(j+1) - f_ref(j));
                    S_ref_interp[i] = (1.0 - t) * S_ref(j) + t * S_ref(j+1);
                } else {
                    S_ref_interp[i] = S_ref(f_ref.size()-1);
                }
            }

            // Write row to CSV
            ofs << block_count << "," << rec.time << ","
                << Hs << "," << Fp << ","
                << pm.alpha << "," << pm.fp << "," << pm.cost;
            for (int i = 0; i < Nfreq; i++) {
                ofs << "," << S_est[i] << "," << S_ref_interp[i];
            }
            ofs << "\n";
        }
    });

    ofs.close();
    std::cout << "Finished " << filename
              << " with " << block_count
              << " spectra → " << outname << "\n\n";
}

// ============================
// Main
// ============================
int main() {
    float dt = 1.0f / 240.0f;
    std::cout << "Standalone spectrum simulation (IMU → estimator vs ref model)\n";

    // gather wave_data files
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

    for (const auto &fname : files) {
        process_wave_file(fname, dt);
    }
    return 0;
}
