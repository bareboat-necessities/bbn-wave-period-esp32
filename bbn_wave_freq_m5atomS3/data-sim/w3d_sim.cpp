#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>   // for std::clamp
#include <string>
#include <vector>
#include <map>
#include <random>

#define EIGEN_NON_ARDUINO

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double g_std = 9.80665;     // standard gravity acceleration m/s²
const double MAG_DELAY_SEC = 5.0; // delay before enabling magnetometer

const double FAIL_ERR_LIMIT_PERCENT_HIGH = 11.50;
const double FAIL_ERR_LIMIT_PERCENT_LOW  = 11.50;

// Global variable set from command line
double R_S_base_global = 1.9;   // default

// RMS window length [s]
constexpr double RMS_WINDOW_SEC = 60.0;

#include "Kalman3D_Wave.h"     // Kalman3D_Wave filter
#include "WaveFilesSupport.h"  // file reader/parser + naming
#include "FrameConversions.h"  // coordinate & quaternion conversions + MagSim_WMM

using Eigen::Vector3d;
using Eigen::Quaterniond;

// Inline RMS accumulator (double)
class RMSReport {
public:
    inline void add(double value) {
        sum_sq_ += value * value;
        count_++;
    }
    inline double rms() const {
        if (count_ == 0) return NAN;
        return std::sqrt(sum_sq_ / static_cast<double>(count_));
    }
private:
    double sum_sq_ = 0.0;
    size_t count_ = 0;
};

// Noise model (accel & gyro noisy by default; disable with --no-noise)
bool add_noise = true;

struct NoiseModel {
    std::default_random_engine rng;
    std::normal_distribution<double> dist;
    Vector3d bias;
};

NoiseModel make_noise_model(double sigma, double bias_range, unsigned seed) {
    NoiseModel m{std::default_random_engine(seed),
                 std::normal_distribution<double>(0.0, sigma),
                 Vector3d::Zero()};
    std::uniform_real_distribution<double> ub(-bias_range, bias_range);
    m.bias = Vector3d(ub(m.rng), ub(m.rng), ub(m.rng));
    return m;
}

Vector3d apply_noise(const Vector3d& v, NoiseModel& m) {
    return v - m.bias + Vector3d(m.dist(m.rng), m.dist(m.rng), m.dist(m.rng));
}

// Example test cases
const std::vector<WaveParameters> waveParamsList = {
    {3.0,   0.27, (M_PI/3.0), 25.0},
    {5.7,   1.5,  (M_PI/1.5), 25.0},
    {8.5,   4.0,  (M_PI/6.0), 25.0},
    {11.4,  8.5,  (M_PI/2.5), 25.0}
};

// R_S law now always uses global
inline double R_S_law(double Tp, double T_p_base = 8.5) {
    return R_S_base_global * std::pow(Tp/T_p_base, 1.0 / 3.0);
}

struct TuningIMU {
    double tau_eff, sigma_a_eff, R_S_eff;
};

// Build tuning_map once using current R_S_base_global
inline std::map<WaveType, std::vector<TuningIMU>> make_tuning_map() {
    return {
        { WaveType::JONSWAP, {
            { 0.461907, 0.488792, R_S_law(3.0)  },
            { 0.875139, 0.993007, R_S_law(5.7)  },
            { 1.314262, 1.344122, R_S_law(8.5)  },
            { 1.757230, 1.709363, R_S_law(11.4) }
        }},
        { WaveType::PMSTOKES, {
            { 0.316639, 0.563192, R_S_law(3.0)  },
            { 0.599911, 1.177536, R_S_law(5.7)  },
            { 0.900931, 1.607994, R_S_law(8.5)  },
            { 1.204587, 2.053963, R_S_law(11.4) }
        }}
    };
}

int wave_index_from_height(double height) {
    for (size_t i = 0; i < waveParamsList.size(); i++) {
        if (std::abs(waveParamsList[i].height - height) < 1e-6) {
            return static_cast<int>(i);
        }
    }
    return -1; // not found
}

void process_wave_file(const std::string &filename, double dt, bool with_mag,
    const std::map<WaveType, std::vector<TuningIMU>>& tuning_map) {
    auto parsed = WaveFileNaming::parse_to_params(filename);
    if (!parsed) return;
    auto [kind, type, wp] = *parsed;
    if (kind != FileKind::Data) return;
    if (!(type == WaveType::JONSWAP || type == WaveType::PMSTOKES)) return;

    int wave_idx = wave_index_from_height(wp.height);
    if (wave_idx < 0) {
        std::cerr << "No tuning found for height=" << wp.height << "\n";
        return;
    }
    const auto &tune = tuning_map.at(type).at(wave_idx);

    std::cout << "Processing " << filename
              << " (" << EnumTraits<WaveType>::to_string(type)
              << ") with R_S_base=" << R_S_base_global
              << " -> R_S_eff=" << tune.R_S_eff << "\n";

    std::cout << "Using tuning for "
              << EnumTraits<WaveType>::to_string(type)
              << " wave index " << wave_idx
              << ": tau=" << tune.tau_eff
              << ", sigma_a=" << tune.sigma_a_eff
              << ", R_S=" << tune.R_S_eff << "\n";

    WaveDataCSVReader reader(filename);

    const Vector3d sigma_a(0.04, 0.04, 0.04);
    const Vector3d sigma_g(0.00134, 0.00134, 0.00134);
    const Vector3d sigma_m(0.3, 0.3, 0.3);
    Kalman3D_Wave<double, true, true> mekf(sigma_a, sigma_g, sigma_m);

    // Configure filter using selected tuning parameters
    mekf.set_aw_time_constant(tune.tau_eff * 2.0);

    // sigma_a_eff was scalar in logs — treat isotropic per axis
    Eigen::Vector3d std_aw = Eigen::Vector3d::Constant(tune.sigma_a_eff);
    mekf.set_aw_stationary_std(std_aw * 1.3);

    // R_S_eff is scalar too — isotropic pseudo-measurement noise
    Eigen::Vector3d sigma_S = Eigen::Vector3d::Constant(tune.R_S_eff);
    mekf.set_RS_noise(sigma_S);

    const Vector3d mag_world_a = MagSim_WMM::mag_world_aero();

    // Noise models (fixed params, seeded differently)
    static NoiseModel accel_noise = make_noise_model(0.03, 0.02, 1234);
    static NoiseModel gyro_noise  = make_noise_model(0.001, 0.0004, 5678);

    bool first = true;
    bool mag_enabled = false;
    int iter = 0;

    // build outname (legacy scheme)
    std::string outname = filename;
    auto pos_prefix = outname.find("wave_data_");
    if (pos_prefix != std::string::npos) {
        outname.replace(pos_prefix, std::string("wave_data_").size(), "w3d_");
    }
    auto pos_ext = outname.rfind(".csv");
    if (pos_ext != std::string::npos) {
        outname.insert(pos_ext, with_mag ? "_w3d" : "_w3d_nomag");
    } else {
        outname += with_mag ? "_w3d.csv" : "_w3d_nomag.csv";
    }

    std::ofstream ofs(outname);
    ofs << "time, ... (headers unchanged) ... \n";  // keep your CSV headers

    // RMS accumulators (for last 60 s)
    std::vector<double> errs_z, errs_roll, errs_pitch, errs_yaw, errs_angle;

    reader.for_each_record([&](const Wave_Data_Sample &rec) {
        iter++;

        Vector3d acc_b(rec.imu.acc_bx, rec.imu.acc_by, rec.imu.acc_bz);
        Vector3d gyr_b(rec.imu.gyro_x, rec.imu.gyro_y, rec.imu.gyro_z);

        // Apply optional noise/bias
        Vector3d acc_noisy = acc_b;
        Vector3d gyr_noisy = gyr_b;
        if (add_noise) {
            acc_noisy = apply_noise(acc_b, accel_noise);
            gyr_noisy = apply_noise(gyr_b, gyro_noise);
        }

        Vector3d acc_f = zu_to_ned(acc_noisy);
        Vector3d gyr_f = zu_to_ned(gyr_noisy);

        // Reference Euler (nautical ENU/Z-up)
        double r_ref_out = rec.imu.roll_deg;
        double p_ref_out = rec.imu.pitch_deg;
        double y_ref_out = rec.imu.yaw_deg;

        // Simulated magnetometer
        Vector3d mag_f(0,0,0);
        if (with_mag) {
            Vector3d mag_b_enu =
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

        // Estimated quaternion → nautical Euler
        auto coeffs = mekf.quaternion().coeffs();
        Quaterniond q(coeffs(3), coeffs(0), coeffs(1), coeffs(2));
        double r_est_a, p_est_a, y_est_a;
        quat_to_euler_aero(q, r_est_a, p_est_a, y_est_a);
        double r_est = r_est_a, p_est = p_est_a, y_est = y_est_a;
        aero_to_nautical(r_est, p_est, y_est);

        // Reference and estimated kinematics
        Vector3d disp_ref(rec.wave.disp_x, rec.wave.disp_y, rec.wave.disp_z);
        Vector3d vel_ref (rec.wave.vel_x,  rec.wave.vel_y,  rec.wave.vel_z);
        Vector3d acc_ref (rec.wave.acc_x,  rec.wave.acc_y,  rec.wave.acc_z);

        Vector3d disp_est = ned_to_zu(mekf.get_position());
        Vector3d vel_est  = ned_to_zu(mekf.get_velocity());
        Vector3d acc_est  = ned_to_zu(mekf.get_world_accel());

        Vector3d disp_err = disp_est - disp_ref;
        Vector3d vel_err  = vel_est  - vel_ref;
        Vector3d acc_err  = acc_est  - acc_ref;

        Quaterniond q_est_naut =
            Eigen::AngleAxisd(r_est*M_PI/180.0, Vector3d::UnitX()) *
            Eigen::AngleAxisd(p_est*M_PI/180.0, Vector3d::UnitY()) *
            Eigen::AngleAxisd(y_est*M_PI/180.0, Vector3d::UnitZ());
        Quaterniond q_ref =
            Eigen::AngleAxisd(r_ref_out*M_PI/180.0, Vector3d::UnitX()) *
            Eigen::AngleAxisd(p_ref_out*M_PI/180.0, Vector3d::UnitY()) *
            Eigen::AngleAxisd(y_ref_out*M_PI/180.0, Vector3d::UnitZ());
        Quaterniond q_err = q_ref.conjugate() * q_est_naut.normalized();
        double angle_err = 2.0 * std::acos(std::clamp(q_err.w(), -1.0, 1.0)) * 180.0/M_PI;

        // Estimated biases
        Vector3d bacc_est = mekf.get_acc_bias();
        Vector3d bgyr_est = mekf.gyroscope_bias();

        // Store errors
        errs_z.push_back(disp_err.z());
        errs_roll.push_back(r_est - r_ref_out);
        errs_pitch.push_back(p_est - p_ref_out);
        errs_yaw.push_back(y_est - y_ref_out);
        errs_angle.push_back(angle_err);

        // Stream row to CSV (unchanged except types)
        ofs << rec.time << "," ... << "\n"; // keep same fields
    });

    ofs.close();
    std::cout << "Wrote " << outname << "\n";

    // RMS summary for last 60 s
    int N_last = static_cast<int>(60.0 / dt);
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

        double z_rms = rms_z.rms();
        double z_rms_pct = 100.0 * z_rms / wp.height;

        std::cout << "=== Last 60 s RMS summary for " << filename << " ===\n";
        std::cout << "Z RMS = " << z_rms
                  << " m (" << z_rms_pct
                  << "% of Hs=" << wp.height << ")\n";
        std::cout << "Angles RMS (deg): "
                  << "Roll=" << rms_roll.rms()
                  << ", Pitch=" << rms_pitch.rms()
                  << ", Yaw=" << rms_yaw.rms() << "\n";
        std::cout << "Absolute angle error RMS (deg): "
                  << rms_ang.rms() << "\n";
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

int main(int argc, char* argv[]) {
    double dt = 1.0 / 240.0;

    bool with_mag = true;
    add_noise = true;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--nomag") with_mag = false;
        else if (arg == "--no-noise") add_noise = false;
        else if (arg.rfind("--rs-base=", 0) == 0) {
            try {
                R_S_base_global = std::stod(arg.substr(10));
            } catch (...) {
                std::cerr << "Invalid value for --rs-base\n";
                return EXIT_FAILURE;
            }
        }
    }

    // Build tuning map once with the chosen R_S_base
    auto tuning_map = make_tuning_map();

    std::cout << "Simulation starting with_mag=" << (with_mag ? "true" : "false")
              << ", mag_delay=" << MAG_DELAY_SEC << " sec"
              << ", noise=" << (add_noise ? "true" : "false")
              << ", R_S_base=" << R_S_base_global << "\n";

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
        process_wave_file(fname, dt, with_mag, tuning_map);
    }
    return 0;
}
