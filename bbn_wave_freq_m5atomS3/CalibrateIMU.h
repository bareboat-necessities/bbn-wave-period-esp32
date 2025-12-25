// imu_ellipsoid_calibration.h
// Header-only, heap-free (fixed-size) ellipsoid calibration for ESP32-S3 + Eigen.
// Calibrates:
//   - Accelerometer: bias + 3x3 soft matrix (scale + cross-axis) by fitting to ||M(a-b)|| = g (auto-detected units)
//   - Magnetometer: hard-iron + soft-iron by fitting to ||M(m-b)|| = r (r estimated; units don’t matter)
//   - Gyro: stillness-based bias estimate (ellipsoid is not the right model without a rotation reference)
//
// Usage:
//   1) Accumulate up to 300 samples (acc, mag) while rotating through many orientations.
//   2) Optionally accumulate still gyro samples for bias.
//   3) Call calibrate().
//   4) Apply:
//        a_cal = acc.M * (a_raw - acc.bias)
//        m_cal = mag.M * (m_raw - mag.bias)
//        w_cal = w_raw - gyro_bias
//
// Notes:
//   - Includes automatic accel unit detection (g, m/s^2, mg, or unknown -> uses median norm as radius)
//   - Includes coverage/spread gating so it won’t fit on a “flat” dataset.

#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
#else
  #include <ArduinoEigenDense.h>
#endif

namespace imu_cal {

template <typename T>
static inline T clamp(T x, T lo, T hi) { return std::min(hi, std::max(lo, x)); }

template <typename T>
static inline T huber_weight(T r, T k) {
  const T a = std::abs(r);
  if (a <= k) return T(1);
  return k / a;
}

// -----------------------------------------------------------------------------
// Fixed-capacity sample buffer (no heap)
// -----------------------------------------------------------------------------
template <typename T, int MaxN>
struct SampleBuffer3 {
  using Vec3 = Eigen::Matrix<T,3,1>;
  Vec3 data[MaxN];
  int  n = 0;

  void clear() { n = 0; }
  int  size() const { return n; }

  bool push(const Vec3& v) {
    if (n >= MaxN) return false;
    data[n++] = v;
    return true;
  }
  const Vec3& operator[](int i) const { return data[i]; }
};

// -----------------------------------------------------------------------------
// Norm stats: median and MAD (robust). Heap-free quickselect.
// -----------------------------------------------------------------------------
template <typename T, int MaxN>
struct NormStats {
  T norms[MaxN];
  int n = 0;

  void clear() { n = 0; }
  int  size() const { return n; }
  bool add_norm(T x) {
    if (n >= MaxN) return false;
    norms[n++] = x;
    return true;
  }

  template <typename Vec3>
  bool add_vec(const Vec3& v) { return add_norm(T(v.norm())); }

  static T quickselect(T* a, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
      const T pivot = a[(lo + hi) >> 1];
      int i = lo, j = hi;
      while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) { std::swap(a[i], a[j]); i++; j--; }
      }
      if (k <= j) hi = j;
      else if (k >= i) lo = i;
      else return a[k];
    }
    return a[lo];
  }

  T median() const {
    if (n <= 0) return T(0);
    T tmp[MaxN];
    for (int i=0;i<n;i++) tmp[i] = norms[i];
    const int k = n >> 1;
    T m = quickselect(tmp, n, k);
    if ((n & 1) == 0) {
      T m2 = quickselect(tmp, n, k - 1);
      return T(0.5) * (m + m2);
    }
    return m;
  }

  T mad() const {
    if (n <= 0) return T(0);
    const T m = median();
    T tmp[MaxN];
    for (int i=0;i<n;i++) tmp[i] = std::abs(norms[i] - m);
    const int k = n >> 1;
    T d = quickselect(tmp, n, k);
    if ((n & 1) == 0) {
      T d2 = quickselect(tmp, n, k - 1);
      return T(0.5) * (d + d2);
    }
    return d;
  }
};

// -----------------------------------------------------------------------------
// Simple spherical coverage bins: (elevation, azimuth) quantized.
// This is cheap and works well as a “do we have enough orientation diversity?” gate.
// -----------------------------------------------------------------------------
template <typename T, int ElevBins, int AzimBins>
struct CoverageBins {
  bool bins[ElevBins * AzimBins];

  void clear() {
    for (int i=0;i<ElevBins*AzimBins;i++) bins[i] = false;
  }

  static inline T rad2deg(T r) { return r * T(57.29577951308232); }

  // v must be non-zero; uses direction only.
  void add_dir(const Eigen::Matrix<T,3,1>& v) {
    const T n = v.norm();
    if (n < T(1e-6)) return;
    const Eigen::Matrix<T,3,1> u = v / n;

    // elevation: 0..180 (theta), azimuth: -180..180
    const T theta = rad2deg(std::acos(clamp(u.z(), T(-1), T(1))));   // 0..180
    const T phi   = rad2deg(std::atan2(u.y(), u.x()));              // -180..180

    int ti = int((theta / T(180)) * T(ElevBins));
    ti = std::min(std::max(ti, 0), ElevBins-1);

    T phi01 = (phi + T(180)) / T(360); // 0..1
    int pi = int(phi01 * T(AzimBins));
    pi = std::min(std::max(pi, 0), AzimBins-1);

    bins[ti*AzimBins + pi] = true;
  }

  int count() const {
    int c = 0;
    for (int i=0;i<ElevBins*AzimBins;i++) if (bins[i]) c++;
    return c;
  }
};

// -----------------------------------------------------------------------------
// Unit detection (accel) and heuristic mag classification (mostly for thresholds)
// -----------------------------------------------------------------------------
enum class AccelUnits : uint8_t { G, MPS2, MG, UNKNOWN };
enum class MagUnits   : uint8_t { UT_LIKE, NORMALIZED, COUNTS, UNKNOWN };

template <typename T>
static inline AccelUnits detect_accel_units(T med_norm) {
  if (med_norm > T(0.6) && med_norm < T(1.6))   return AccelUnits::G;      // ~1g
  if (med_norm > T(6.0) && med_norm < T(13.0))  return AccelUnits::MPS2;   // ~9.8
  if (med_norm > T(600) && med_norm < T(1600))  return AccelUnits::MG;     // ~1000 mg
  return AccelUnits::UNKNOWN;
}

template <typename T>
static inline T accel_radius_for_units(AccelUnits u) {
  switch (u) {
    case AccelUnits::G:    return T(1);
    case AccelUnits::MPS2: return T(9.80665);
    case AccelUnits::MG:   return T(1000);
    default:               return T(-1);
  }
}

template <typename T>
static inline MagUnits detect_mag_units(T med_norm) {
  if (med_norm > T(5) && med_norm < T(200))      return MagUnits::UT_LIKE;     // µT-ish
  if (med_norm > T(0.2) && med_norm < T(2.5))    return MagUnits::NORMALIZED;  // ~unit
  if (med_norm > T(200) && med_norm < T(100000)) return MagUnits::COUNTS;      // raw counts
  return MagUnits::UNKNOWN;
}

// -----------------------------------------------------------------------------
// Ellipsoid fitter (LM / Gauss-Newton), heap-free fixed-size matrices.
// Model: || L (x - b) || ~= r
// L is lower-triangular with positive diagonal via exp(logdiag).
// Parameters:
//   b(3), offdiag(3) = [l10, l20, l21], logdiag(3) = [log l00, log l11, log l22], and optional r.
// -----------------------------------------------------------------------------
template <typename T, int MaxN>
class EllipsoidFitter {
public:
  using Vec3 = Eigen::Matrix<T,3,1>;
  using Mat3 = Eigen::Matrix<T,3,3>;

  struct Result {
    bool ok = false;
    Vec3 bias = Vec3::Zero();
    Mat3 M = Mat3::Identity(); // apply y = M*(x-bias)
    T radius = T(1);
    T rms = T(0);
    int iters = 0;
  };

  void clear() {
    samples_.clear();
    b_.setZero();
    off_.setZero();
    logd_.setZero();
    r_ = T(1);
    estimate_radius_ = false;
    r_fixed_ = T(1);
    max_iters_ = 15;
    lambda0_ = T(1e-3);
    huber_k_ = T(0.05);
  }

  bool addSample(const Vec3& x) { return samples_.push(x); }
  int  size() const { return samples_.size(); }

  void setEstimateRadius(bool en) { estimate_radius_ = en; }
  void setFixedRadius(T r) { r_fixed_ = r; r_ = r; estimate_radius_ = false; }
  void setMaxIters(int it) { max_iters_ = it; }
  void setLambda0(T lam) { lambda0_ = lam; }
  void setHuberK(T k) { huber_k_ = std::max(T(1e-9), k); }

  // Initialize b as mean, and L scale so average norm matches target radius (if fixed).
  void initializeFromData() {
    const int N = samples_.size();
    if (N <= 0) return;

    Vec3 mean = Vec3::Zero();
    for (int i=0;i<N;i++) mean += samples_[i];
    mean /= T(N);
    b_ = mean;

    T avg_norm = T(0);
    for (int i=0;i<N;i++) avg_norm += (samples_[i] - b_).norm();
    avg_norm = (avg_norm > T(1e-9)) ? (avg_norm / T(N)) : T(1);

    if (estimate_radius_) {
      r_ = avg_norm;
      logd_.setZero(); // identity
    } else {
      r_ = r_fixed_;
      const T s = std::max(T(1e-6), r_fixed_ / avg_norm);
      logd_ = Vec3::Constant(std::log(s));
    }
    off_.setZero();
  }

  Result fit() {
    Result out;
    const int N = samples_.size();
    if (N < 12) return out;

    if (estimate_radius_) {
      out = fit_impl_10_();
    } else {
      out = fit_impl_9_();
    }
    return out;
  }

private:
  Mat3 buildL_() const {
    Mat3 L = Mat3::Zero();
    L(0,0) = std::exp(logd_(0));
    L(1,0) = off_(0);
    L(1,1) = std::exp(logd_(1));
    L(2,0) = off_(1);
    L(2,1) = off_(2);
    L(2,2) = std::exp(logd_(2));
    return L;
  }

  Result finalize_result_() const {
    Result out;
    out.ok = true;
    out.bias = b_;
    out.M = buildL_();
    out.radius = r_;

    // RMS
    const Mat3 L = out.M;
    T sum2 = T(0);
    int used = 0;
    for (int i=0;i<samples_.size();i++) {
      const Vec3 y = L * (samples_[i] - out.bias);
      const T s = y.norm();
      if (s < T(1e-9)) continue;
      const T e = s - out.radius;
      sum2 += e*e;
      used++;
    }
    out.rms = (used>0) ? std::sqrt(sum2 / T(used)) : T(0);
    return out;
  }

  Result fit_impl_9_() {
    using Mat9 = Eigen::Matrix<T,9,9>;
    using Vec9 = Eigen::Matrix<T,9,1>;

    Result out;
    T lambda = lambda0_;

    for (int it=0; it<max_iters_; ++it) {
      Mat9 JTJ = Mat9::Zero();
      Vec9 JTr = Vec9::Zero();
      T cost = T(0);
      int used = 0;

      const Mat3 L = buildL_();

      for (int i=0;i<samples_.size();i++) {
        const Vec3 x = samples_[i];
        const Vec3 v = x - b_;
        const Vec3 y = L * v;
        const T s = y.norm();
        if (s < T(1e-9)) continue;

        const T e = s - r_fixed_;
        const T w = huber_weight(e, huber_k_);
        const T we = w * e;
        cost += we * we;
        used++;

        const Vec3 u = y / s;

        // Jacobians
        const Vec3 j_b = -(L.transpose() * u);

        const T j_l10 = u(1) * v(0);
        const T j_l20 = u(2) * v(0);
        const T j_l21 = u(2) * v(1);

        const T j_d0 = (u(0) * v(0)) * L(0,0);
        const T j_d1 = (u(1) * v(1)) * L(1,1);
        const T j_d2 = (u(2) * v(2)) * L(2,2);

        Vec9 J;
        J << j_b(0), j_b(1), j_b(2),
             j_l10, j_l20, j_l21,
             j_d0, j_d1, j_d2;

        const T ww = w*w;
        JTJ.noalias() += ww * (J * J.transpose());
        JTr.noalias() += ww * (J * e);
      }

      if (used < 9) break;

      Mat9 A = JTJ;
      for (int k=0;k<9;k++) A(k,k) += lambda;

      Eigen::LDLT<Mat9> ldlt(A);
      if (ldlt.info() != Eigen::Success) break;
      const Vec9 dx = ldlt.solve(-JTr);
      if (ldlt.info() != Eigen::Success) break;

      // Save
      const Vec3 b0 = b_;
      const Vec3 off0 = off_;
      const Vec3 logd0 = logd_;

      // Apply
      b_ += dx.template segment<3>(0);
      off_(0) += dx(3); off_(1) += dx(4); off_(2) += dx(5);
      logd_(0) += dx(6); logd_(1) += dx(7); logd_(2) += dx(8);
      for (int k=0;k<3;k++) logd_(k) = clamp(logd_(k), T(-6), T(6));
      r_ = r_fixed_;

      // Evaluate
      const Mat3 Lnew = buildL_();
      T new_cost = T(0);
      int used2 = 0;
      for (int i=0;i<samples_.size();i++) {
        const Vec3 y = Lnew * (samples_[i] - b_);
        const T s = y.norm();
        if (s < T(1e-9)) continue;
        const T e = s - r_fixed_;
        const T w = huber_weight(e, huber_k_);
        new_cost += (w*e)*(w*e);
        used2++;
      }
      if (used2 < 9) new_cost = std::numeric_limits<T>::infinity();

      if (new_cost < cost) {
        lambda = std::max(lambda * T(0.3), T(1e-12));
        out.iters = it+1;
        if (dx.norm() < T(1e-6)) break;
      } else {
        // reject
        b_ = b0; off_ = off0; logd_ = logd0;
        lambda = std::min(lambda * T(10), T(1e6));
      }
    }

    out = finalize_result_();
    out.radius = r_fixed_;
    return out;
  }

  Result fit_impl_10_() {
    using Mat10 = Eigen::Matrix<T,10,10>;
    using Vec10 = Eigen::Matrix<T,10,1>;

    Result out;
    T lambda = lambda0_;

    for (int it=0; it<max_iters_; ++it) {
      Mat10 JTJ = Mat10::Zero();
      Vec10 JTr = Vec10::Zero();
      T cost = T(0);
      int used = 0;

      const Mat3 L = buildL_();

      for (int i=0;i<samples_.size();i++) {
        const Vec3 x = samples_[i];
        const Vec3 v = x - b_;
        const Vec3 y = L * v;
        const T s = y.norm();
        if (s < T(1e-9)) continue;

        const T e = s - r_;
        const T w = huber_weight(e, huber_k_);
        const T we = w * e;
        cost += we * we;
        used++;

        const Vec3 u = y / s;

        const Vec3 j_b = -(L.transpose() * u);

        const T j_l10 = u(1) * v(0);
        const T j_l20 = u(2) * v(0);
        const T j_l21 = u(2) * v(1);

        const T j_d0 = (u(0) * v(0)) * L(0,0);
        const T j_d1 = (u(1) * v(1)) * L(1,1);
        const T j_d2 = (u(2) * v(2)) * L(2,2);

        Vec10 J;
        J << j_b(0), j_b(1), j_b(2),
             j_l10, j_l20, j_l21,
             j_d0, j_d1, j_d2,
             T(-1); // de/dr

        const T ww = w*w;
        JTJ.noalias() += ww * (J * J.transpose());
        JTr.noalias() += ww * (J * e);
      }

      if (used < 10) break;

      Mat10 A = JTJ;
      for (int k=0;k<10;k++) A(k,k) += lambda;

      Eigen::LDLT<Mat10> ldlt(A);
      if (ldlt.info() != Eigen::Success) break;
      const Vec10 dx = ldlt.solve(-JTr);
      if (ldlt.info() != Eigen::Success) break;

      // Save
      const Vec3 b0 = b_;
      const Vec3 off0 = off_;
      const Vec3 logd0 = logd_;
      const T r0 = r_;

      // Apply
      b_ += dx.template segment<3>(0);
      off_(0) += dx(3); off_(1) += dx(4); off_(2) += dx(5);
      logd_(0) += dx(6); logd_(1) += dx(7); logd_(2) += dx(8);
      for (int k=0;k<3;k++) logd_(k) = clamp(logd_(k), T(-6), T(6));
      r_ += dx(9);
      r_ = std::max(T(1e-6), r_);

      // Evaluate
      const Mat3 Lnew = buildL_();
      T new_cost = T(0);
      int used2 = 0;
      for (int i=0;i<samples_.size();i++) {
        const Vec3 y = Lnew * (samples_[i] - b_);
        const T s = y.norm();
        if (s < T(1e-9)) continue;
        const T e = s - r_;
        const T w = huber_weight(e, huber_k_);
        new_cost += (w*e)*(w*e);
        used2++;
      }
      if (used2 < 10) new_cost = std::numeric_limits<T>::infinity();

      if (new_cost < cost) {
        lambda = std::max(lambda * T(0.3), T(1e-12));
        out.iters = it+1;
        if (dx.norm() < T(1e-6)) break;
      } else {
        // reject
        b_ = b0; off_ = off0; logd_ = logd0; r_ = r0;
        lambda = std::min(lambda * T(10), T(1e6));
      }
    }

    out = finalize_result_();
    return out;
  }

  SampleBuffer3<T, MaxN> samples_;

  bool estimate_radius_ = false;
  T r_fixed_ = T(1);

  int max_iters_ = 15;
  T lambda0_ = T(1e-3);
  T huber_k_ = T(0.05);

  Vec3 b_ = Vec3::Zero();
  Vec3 off_ = Vec3::Zero();   // l10, l20, l21
  Vec3 logd_ = Vec3::Zero();  // log diag
  T    r_ = T(1);
};

// -----------------------------------------------------------------------------
// Gyro bias estimator (stillness-based). Heap-free.
// -----------------------------------------------------------------------------
template <typename T>
class GyroBiasEstimator {
public:
  using Vec3 = Eigen::Matrix<T,3,1>;
  void reset() { n_ = 0; mean_.setZero(); m2_.setZero(); }

  void addStillSample(const Vec3& w) {
    n_++;
    const Vec3 delta = w - mean_;
    mean_ += delta / T(n_);
    const Vec3 delta2 = w - mean_;
    m2_ += delta.cwiseProduct(delta2);
  }

  int count() const { return n_; }
  Vec3 bias() const { return mean_; }
  Vec3 variance() const { return (n_ >= 2) ? (m2_ / T(n_ - 1)) : Vec3::Zero(); }
  bool ready(int min_samples) const { return n_ >= min_samples; }

private:
  int n_ = 0;
  Vec3 mean_ = Vec3::Zero();
  Vec3 m2_   = Vec3::Zero();
};

// -----------------------------------------------------------------------------
// Auto unit detection + threshold selection + spread/coverage gating
// -----------------------------------------------------------------------------
template <typename T, int MaxN>
struct AutoConfig {
  using Vec3 = Eigen::Matrix<T,3,1>;

  NormStats<T, MaxN> acc_stats;
  NormStats<T, MaxN> mag_stats;

  AccelUnits acc_units = AccelUnits::UNKNOWN;
  MagUnits   mag_units = MagUnits::UNKNOWN;

  T acc_radius = T(1);
  T acc_huber_k = T(0.05);
  T mag_huber_k = T(0.5);

  // Spread gating thresholds (tunable defaults)
  int min_samples_acc = 80;
  int min_samples_mag = 80;

  // Coverage bins: 9 elevation x 18 azimuth = 162 bins
  // Require a decent fraction to be hit (roughly “rotated around”).
  int min_bins_acc = 36; // ~20-25% of bins
  int min_bins_mag = 54; // mag usually needs more coverage

  // Norm gating for coverage (reject norms far from median)
  T acc_norm_gate_sigma = T(4); // accept if |norm - med| <= sigma * MAD + 0.08*med
  T mag_norm_gate_sigma = T(4);

  void finalize() {
    const T acc_med = acc_stats.median();
    const T mag_med = mag_stats.median();

    acc_units = detect_accel_units(acc_med);
    mag_units = detect_mag_units(mag_med);

    T r = accel_radius_for_units<T>(acc_units);
    if (r <= T(0)) {
      // Unknown accel units: keep native scale; target median norm.
      r = (acc_med > T(1e-6)) ? acc_med : T(1);
    }
    acc_radius = r;

    const T acc_mad = acc_stats.mad();
    acc_huber_k = std::max(T(0.02) * acc_radius, T(3) * acc_mad);

    const T mag_mad = mag_stats.mad();
    const T mag_scale = (mag_med > T(1e-6)) ? mag_med : T(1);
    mag_huber_k = std::max(T(0.03) * mag_scale, T(3) * mag_mad);
  }

  template <typename Fitter>
  void apply_accel(Fitter& f) const {
    f.setEstimateRadius(false);
    f.setFixedRadius(acc_radius);
    f.setHuberK(acc_huber_k);
  }

  template <typename Fitter>
  void apply_mag(Fitter& f) const {
    f.setEstimateRadius(true);
    f.setHuberK(mag_huber_k);
  }

  // Coverage/spread check using binned directions with norm gating.
  template <int ElevBins, int AzimBins>
  bool accel_has_coverage(const SampleBuffer3<T, MaxN>& acc_samples) const {
    if (acc_samples.size() < min_samples_acc) return false;

    CoverageBins<T, ElevBins, AzimBins> cov; cov.clear();
    const T med = acc_stats.median();
    const T mad = acc_stats.mad();
    const T gate = acc_norm_gate_sigma * mad + T(0.08) * med;

    for (int i=0;i<acc_samples.size();i++) {
      const Vec3 a = acc_samples[i];
      const T n = a.norm();
      if (std::abs(n - med) > gate) continue;
      cov.add_dir(a);
    }
    return cov.count() >= min_bins_acc;
  }

  template <int ElevBins, int AzimBins>
  bool mag_has_coverage(const SampleBuffer3<T, MaxN>& mag_samples) const {
    if (mag_samples.size() < min_samples_mag) return false;

    CoverageBins<T, ElevBins, AzimBins> cov; cov.clear();
    const T med = mag_stats.median();
    const T mad = mag_stats.mad();
    const T gate = mag_norm_gate_sigma * mad + T(0.15) * med;

    for (int i=0;i<mag_samples.size();i++) {
      const Vec3 m = mag_samples[i];
      const T n = m.norm();
      if (std::abs(n - med) > gate) continue;
      cov.add_dir(m);
    }
    return cov.count() >= min_bins_mag;
  }
};

// -----------------------------------------------------------------------------
// One-stop IMU calibrator: accumulates up to 300 samples and runs both fits.
// -----------------------------------------------------------------------------
template <typename T, int MaxN>
class ImuEllipsoidCalibrator {
public:
  using Vec3 = Eigen::Matrix<T,3,1>;
  using Mat3 = Eigen::Matrix<T,3,3>;

  struct CalResult {
    bool ok_acc = false;
    bool ok_mag = false;
    bool ok_gyro = false;

    typename EllipsoidFitter<T, MaxN>::Result accel;
    typename EllipsoidFitter<T, MaxN>::Result mag;

    Vec3 gyro_bias = Vec3::Zero();

    AccelUnits accel_units = AccelUnits::UNKNOWN;
    MagUnits   mag_units   = MagUnits::UNKNOWN;
  };

  void reset() {
    acc_fit_.clear();
    mag_fit_.clear();
    gyro_bias_.reset();
    cfg_ = AutoConfig<T, MaxN>();
    acc_samples_.clear();
    mag_samples_.clear();
  }

  // Add raw samples (you decide gating before calling these if you want).
  bool add_accel(const Vec3& a) {
    cfg_.acc_stats.add_vec(a);
    acc_samples_.push(a);
    return acc_fit_.addSample(a);
  }

  bool add_mag(const Vec3& m) {
    cfg_.mag_stats.add_vec(m);
    mag_samples_.push(m);
    return mag_fit_.addSample(m);
  }

  // Gyro stillness samples
  void add_gyro_still(const Vec3& w) { gyro_bias_.addStillSample(w); }

  int accel_count() const { return acc_samples_.size(); }
  int mag_count() const { return mag_samples_.size(); }
  int gyro_still_count() const { return gyro_bias_.count(); }

  AutoConfig<T, MaxN>& config() { return cfg_; }
  const AutoConfig<T, MaxN>& config() const { return cfg_; }

  CalResult calibrate() {
    CalResult out;

    // 1) finalize auto units / thresholds
    cfg_.finalize();
    out.accel_units = cfg_.acc_units;
    out.mag_units = cfg_.mag_units;

    // 2) coverage gating
    // 9x18 bins are a good compromise for embedded.
    const bool acc_ok = cfg_.template accel_has_coverage<9,18>(acc_samples_);
    const bool mag_ok = cfg_.template mag_has_coverage<9,18>(mag_samples_);

    // 3) fit accel
    if (acc_ok) {
      cfg_.apply_accel(acc_fit_);
      acc_fit_.initializeFromData();
      out.accel = acc_fit_.fit();
      out.ok_acc = out.accel.ok;
    }

    // 4) fit mag
    if (mag_ok) {
      cfg_.apply_mag(mag_fit_);
      mag_fit_.initializeFromData();
      out.mag = mag_fit_.fit();
      out.ok_mag = out.mag.ok;
    }

    // 5) gyro bias
    if (gyro_bias_.ready(300)) {
      out.gyro_bias = gyro_bias_.bias();
      out.ok_gyro = true;
    }

    return out;
  }

  // Apply helpers
  static inline Vec3 apply_cal(const typename EllipsoidFitter<T, MaxN>::Result& r, const Vec3& x) {
    return r.M * (x - r.bias);
  }

private:
  AutoConfig<T, MaxN> cfg_;

  // Keep sample buffers for coverage checks
  SampleBuffer3<T, MaxN> acc_samples_;
  SampleBuffer3<T, MaxN> mag_samples_;

  EllipsoidFitter<T, MaxN> acc_fit_;
  EllipsoidFitter<T, MaxN> mag_fit_;
  GyroBiasEstimator<T>     gyro_bias_;
};

} // namespace imu_cal

// -----------------------------------------------------------------------------
// Example (pseudo-usage):
//
// imu_cal::ImuEllipsoidCalibrator<float, 300> cal;
// cal.reset();
//
// while (collecting) {
//   // Read raw IMU in your chosen units
//   Eigen::Vector3f a = acc_raw;
//   Eigen::Vector3f m = mag_raw;
//   Eigen::Vector3f w = gyro_raw;
//
//   cal.add_accel(a);
//   cal.add_mag(m);
//
//   // Stillness detection (you should implement):
//   // if (abs(|a|-median_g) small && |w| small) cal.add_gyro_still(w);
//
//   // Stop when you have ~300 samples and good orientation coverage.
// }
//
// auto res = cal.calibrate();
// if (res.ok_acc) {
//   Eigen::Vector3f a_cal = imu_cal::ImuEllipsoidCalibrator<float,300>::apply_cal(res.accel, acc_raw);
// }
// if (res.ok_mag) {
//   Eigen::Vector3f m_cal = imu_cal::ImuEllipsoidCalibrator<float,300>::apply_cal(res.mag, mag_raw);
// }
// if (res.ok_gyro) {
//   Eigen::Vector3f w_cal = gyro_raw - res.gyro_bias;
// }
// -----------------------------------------------------------------------------
