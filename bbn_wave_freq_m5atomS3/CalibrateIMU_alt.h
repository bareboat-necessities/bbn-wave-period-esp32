#pragma once
// Modern embedded-friendly IMU calibration for Arduino + Eigen.
// Units:
//   accel: m/s^2
//   mag:   uT
//   gyro:  rad/s
//
// Dependencies: ArduinoEigenDense (fixed-size Eigen subset), <cmath>, <stdint.h>

#include <stdint.h>
#include <math.h>
#include <Eigen/Dense>

namespace imu_cal {

// ============================
// Utilities
// ============================
template <typename T>
static inline T clamp(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }

static inline bool finitef(float x) { return isfinite(x); }

template <typename T>
static inline bool isfinite3(const Eigen::Matrix<T,3,1>& v) {
  return finitef((float)v.x()) && finitef((float)v.y()) && finitef((float)v.z());
}

template <typename T>
static inline T sqr(T x) { return x*x; }

template <typename T>
static inline void sort_small(T* a, int n) {
  // Simple insertion sort (n<=400)
  for (int i = 1; i < n; ++i) {
    T key = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > key) { a[j+1] = a[j]; --j; }
    a[j+1] = key;
  }
}

template <typename T>
static inline T median_of_array(T* a, int n) {
  if (n <= 0) return T(0);
  sort_small(a, n);
  if (n & 1) return a[n/2];
  return T(0.5) * (a[n/2 - 1] + a[n/2]);
}

template <typename T>
static inline T robust_mad(T* residuals, int n) {
  // MAD = median(|r - median(r)|)
  if (n <= 0) return T(0);
  T med = median_of_array(residuals, n);
  for (int i = 0; i < n; ++i) residuals[i] = (T)fabs((double)(residuals[i] - med));
  T mad = median_of_array(residuals, n);
  return mad;
}

// ============================
// Fixed-capacity sample buffer
// ============================
template <typename T, int N>
struct SampleBuffer3 {
  using Vec3 = Eigen::Matrix<T,3,1>;

  Vec3 v[N];
  T    tempC[N];
  int  n = 0;

  void clear() { n = 0; }

  bool push(const Vec3& x, T tC = T(0)) {
    if (n >= N) return false;
    if (!isfinite3(x) || !finitef((float)tC)) return false;
    v[n] = x;
    tempC[n] = tC;
    ++n;
    return true;
  }
};

// ============================
// Ellipsoid -> sphere robust fit
// Model: x^T Q x + q^T x + c = 0
// Row d = [x^2 y^2 z^2 2xy 2xz 2yz 2x 2y 2z 1]
// Solve (weighted/trimmed) least squares for p s.t. D p ≈ 1.
// Then recover center b and whitening A such that ||A(x-b)||≈1.
// Then scale A to desired output radius.
// ============================
template <typename T>
struct EllipsoidSphereFit {
  bool ok = false;
  Eigen::Matrix<T,3,1> b = Eigen::Matrix<T,3,1>::Zero();
  Eigen::Matrix<T,3,3> A = Eigen::Matrix<T,3,3>::Identity(); // scaled whitening

  // Diagnostics
  T rms = T(0);     // RMS of (||A(x-b)|| - R_target)
  T median_r = T(0);
  int used = 0;
};

template <typename T>
static inline void build_row_d(const Eigen::Matrix<T,3,1>& x, Eigen::Matrix<T,10,1>& d) {
  const T X=x.x(), Y=x.y(), Z=x.z();
  d << X*X, Y*Y, Z*Z,
       T(2)*X*Y, T(2)*X*Z, T(2)*Y*Z,
       T(2)*X, T(2)*Y, T(2)*Z,
       T(1);
}

template <typename T>
static bool solve_ellipsoid_params_trimmed(
    const Eigen::Matrix<T,3,1>* x, int n,
    const bool* inlier,
    Eigen::Matrix<T,10,1>& p_out,
    T ridge)
{
  using Mat10 = Eigen::Matrix<T,10,10>;
  using Vec10 = Eigen::Matrix<T,10,1>;

  Mat10 H = Mat10::Zero();
  Vec10 g = Vec10::Zero();

  int used = 0;
  for (int i = 0; i < n; ++i) {
    if (inlier && !inlier[i]) continue;
    Vec10 d;
    build_row_d(x[i], d);
    H.noalias() += d * d.transpose();
    g.noalias() += d; // D^T * 1
    ++used;
  }
  if (used < 10) return false;

  H.diagonal().array() += ridge;

  // Solve H p = g
  Eigen::LDLT<Mat10> ldlt(H);
  if (ldlt.info() != Eigen::Success) return false;
  Vec10 p = ldlt.solve(g);
  if (!(p.array().isFinite().all())) return false;

  p_out = p;
  return true;
}

template <typename T>
static EllipsoidSphereFit<T> ellipsoid_to_sphere_robust(
    const Eigen::Matrix<T,3,1>* x, int n,
    T R_target,
    int robust_iters = 3,
    T trim_frac = T(0.15),     // drop worst 15% residuals each iteration
    T ridge = T(1e-6))
{
  EllipsoidSphereFit<T> out;
  if (n < 12) return out;

  bool inlier[400]; // n <= 400 by your constraint
  if (n > 400) return out;
  for (int i = 0; i < n; ++i) inlier[i] = true;

  Eigen::Matrix<T,10,1> p;

  Eigen::Matrix<T,3,1> b = Eigen::Matrix<T,3,1>::Zero();
  Eigen::Matrix<T,3,3> A_unit = Eigen::Matrix<T,3,3>::Identity();

  T residuals[400];

  for (int it = 0; it < robust_iters; ++it) {
    if (!solve_ellipsoid_params_trimmed<T>(x, n, inlier, p, ridge)) return out;

    // Build Q, q, c
    Eigen::Matrix<T,3,3> Q;
    Q << p(0), p(3), p(4),
         p(3), p(1), p(5),
         p(4), p(5), p(2);
    Eigen::Matrix<T,3,1> q;
    q << p(6), p(7), p(8);
    const T c = p(9);

    // b = -0.5 Q^{-1} q
    Eigen::FullPivLU<Eigen::Matrix<T,3,3>> lu(Q);
    if (!lu.isInvertible()) return out;
    b = T(-0.5) * lu.solve(q);

    // k = c + b^T Q b + q^T b  (should be negative)
    const T k = c + b.dot(Q*b) + q.dot(b);
    if (!finitef((float)k) || k >= T(0)) return out;

    Eigen::Matrix<T,3,3> M = Q / (-k);
    M = T(0.5) * (M + M.transpose());

    Eigen::LLT<Eigen::Matrix<T,3,3>> llt(M);
    if (llt.info() != Eigen::Success) return out;

    A_unit = llt.matrixU(); // so that ||A_unit(x-b)|| ~ 1

    // Compute residuals on inliers
    int m = 0;
    for (int i = 0; i < n; ++i) {
      if (!inlier[i]) continue;
      const T r = (A_unit * (x[i] - b)).norm();
      residuals[m++] = r - T(1);
    }
    if (m < 10) return out;

    // Trim worst residuals
    // Use absolute residual ranking by threshold from MAD or by quantile.
    // Here: quantile trim (simple, stable).
    T absr[400];
    for (int i = 0; i < m; ++i) absr[i] = (T)fabs((double)residuals[i]);
    T thr = T(0);

    // Find trim threshold as (1-trim_frac) quantile of abs residuals
    sort_small(absr, m);
    int keep = (int)floor((double)m * (double)(T(1) - trim_frac));
    keep = clamp<int>(keep, 10, m);
    thr = absr[keep - 1];

    // Update inlier mask using threshold on abs residual in original indexing
    int idx = 0;
    for (int i = 0; i < n; ++i) {
      if (!inlier[i]) continue;
      const T r = (A_unit * (x[i] - b)).norm();
      const T e = (T)fabs((double)(r - T(1)));
      // keep those within threshold
      bool keepi = (e <= thr);
      inlier[i] = keepi;
      idx++;
    }
  }

  // Final metrics + scale A to target radius
  Eigen::Matrix<T,3,3> A = A_unit * R_target;

  int used = 0;
  T sum_e2 = 0;
  T radii[400];
  for (int i = 0; i < n; ++i) {
    if (!inlier[i]) continue;
    const T r = (A * (x[i] - b)).norm();
    radii[used] = r;
    const T e = r - R_target;
    sum_e2 += e*e;
    used++;
  }
  if (used < 10) return out;

  T medr = median_of_array(radii, used);
  out.ok = true;
  out.b = b;
  out.A = A;
  out.rms = (T)sqrt((double)(sum_e2 / (T)used));
  out.median_r = medr;
  out.used = used;
  return out;
}

// ============================
// Temperature model: bias(T) = b0 + k*(T - T0)
// Fit per-axis with simple LS.
// ============================
template <typename T>
struct TempBias3 {
  bool ok = false;
  T T0 = T(25);
  Eigen::Matrix<T,3,1> b0 = Eigen::Matrix<T,3,1>::Zero();
  Eigen::Matrix<T,3,1> k  = Eigen::Matrix<T,3,1>::Zero();

  Eigen::Matrix<T,3,1> bias(T tempC) const { return b0 + k * (tempC - T0); }
};

template <typename T, int N>
static TempBias3<T> fit_temp_bias3(const Eigen::Matrix<T,3,1>(&b)[N], const T(&t)[N], int n, T T0) {
  TempBias3<T> out;
  out.T0 = T0;
  if (n < 2) return out;

  T Sx = 0, Sxx = 0;
  Eigen::Matrix<T,3,1> Sy = Eigen::Matrix<T,3,1>::Zero();
  Eigen::Matrix<T,3,1> Sxy = Eigen::Matrix<T,3,1>::Zero();

  for (int i = 0; i < n; ++i) {
    const T x = t[i] - T0;
    Sx += x;
    Sxx += x*x;
    Sy += b[i];
    Sxy += b[i] * x;
  }
  const T det = (T)n * Sxx - Sx*Sx;
  if (fabs((double)det) < 1e-9) return out;

  out.b0 = (Sy * Sxx - Sxy * Sx) / det;
  out.k  = (Sxy * (T)n - Sy * Sx) / det;
  out.ok = true;
  return out;
}

// ============================
// Calibration outputs
// ============================
template <typename T>
struct AccelCalibration {
  bool ok = false;
  T g = T(9.80665);
  Eigen::Matrix<T,3,3> S = Eigen::Matrix<T,3,3>::Identity(); // a_cal = S*(a_raw - bias(T))
  TempBias3<T> biasT;
  T rms_mag = T(0);

  Eigen::Matrix<T,3,1> apply(const Eigen::Matrix<T,3,1>& a_raw, T tempC) const {
    return S * (a_raw - biasT.bias(tempC));
  }
};

template <typename T>
struct MagCalibration {
  bool ok = false;
  Eigen::Matrix<T,3,3> A = Eigen::Matrix<T,3,3>::Identity(); // m_cal = A*(m_raw - b)
  Eigen::Matrix<T,3,1> b = Eigen::Matrix<T,3,1>::Zero();
  T field_uT = T(0);
  T rms = T(0);

  Eigen::Matrix<T,3,1> apply(const Eigen::Matrix<T,3,1>& m_raw) const {
    return A * (m_raw - b);
  }
};

template <typename T>
struct GyroCalibration {
  bool ok = false;
  Eigen::Matrix<T,3,3> S = Eigen::Matrix<T,3,3>::Identity(); // keep identity unless you have a known-rate rig
  TempBias3<T> biasT;

  Eigen::Matrix<T,3,1> apply(const Eigen::Matrix<T,3,1>& w_raw, T tempC) const {
    return S * (w_raw - biasT.bias(tempC));
  }
};

// ============================
// Calibrator: Accel
// Strategy:
//  - You collect ~400 quasi-static accel samples across orientations.
//  - We auto-bin samples by temperature into up to K bins.
//  - For each bin, robust ellipsoid->sphere fit => get center b_bin and S_bin.
//  - Use the best bin (most samples, best rms) for S.
//  - Regress b_bin vs temperature => bias(T).
// ============================
template <typename T, int N, int K_TBINS = 8>
struct AccelCalibrator {
  using Vec3 = Eigen::Matrix<T,3,1>;
  SampleBuffer3<T, N> buf;

  T g = T(9.80665);
  T T0 = T(25);

  // acceptance gates
  T max_gyro_for_static = T(0.35);  // rad/s
  T accel_mag_tol = T(1.0);         // m/s^2, accept if | |a| - g | < tol

  void clear() { buf.clear(); }

  bool addSample(const Vec3& a_raw, const Vec3& w_raw, T tempC) {
    // quasi-static gate: accel magnitude close to g and gyro small
    if (!isfinite3(a_raw) || !isfinite3(w_raw) || !finitef((float)tempC)) return false;
    const T amag = a_raw.norm();
    const T wmag = w_raw.norm();
    if ((T)fabs((double)(amag - g)) > accel_mag_tol) return false;
    if (wmag > max_gyro_for_static) return false;
    return buf.push(a_raw, tempC);
  }

  bool fit(AccelCalibration<T>& out, int robust_iters = 3, T trim_frac = T(0.15)) const {
    if (buf.n < 80) return false;

    // Temperature bins by range
    T tmin = buf.tempC[0], tmax = buf.tempC[0];
    for (int i = 1; i < buf.n; ++i) { tmin = (buf.tempC[i] < tmin ? buf.tempC[i] : tmin); tmax = (buf.tempC[i] > tmax ? buf.tempC[i] : tmax); }
    const T trange = tmax - tmin;
    const T binW = (trange > T(1e-3)) ? (trange / (T)K_TBINS) : T(1);

    Vec3 xbin[K_TBINS][N];
    T    tbin[K_TBINS];
    int  nbin[K_TBINS];
    for (int k = 0; k < K_TBINS; ++k) { nbin[k] = 0; tbin[k] = tmin + (T(k) + T(0.5))*binW; }

    // assign
    for (int i = 0; i < buf.n; ++i) {
      int k = 0;
      if (trange > T(1e-3)) {
        k = (int)floor((double)((buf.tempC[i] - tmin) / binW));
        k = clamp<int>(k, 0, K_TBINS - 1);
      }
      int& nk = nbin[k];
      if (nk < N) xbin[k][nk++] = buf.v[i];
    }

    // Fit each non-empty bin
    Vec3 centers[K_TBINS];
    T temps[K_TBINS];
    int nb = 0;

    // Choose S from the "best" bin (lowest rms, enough samples)
    T best_score = T(1e30);
    Eigen::Matrix<T,3,3> bestS = Eigen::Matrix<T,3,3>::Identity();
    T best_rms = T(0);
    Vec3 best_center = Vec3::Zero();

    for (int k = 0; k < K_TBINS; ++k) {
      if (nbin[k] < 30) continue;
      auto fitk = ellipsoid_to_sphere_robust<T>(xbin[k], nbin[k], g, robust_iters, trim_frac);
      if (!fitk.ok) continue;

      centers[nb] = fitk.b;
      temps[nb]   = tbin[k];
      nb++;

      // score: rms + small penalty for fewer used points
      T score = fitk.rms + T(0.2) * (T(50) / (T)fitk.used);
      if (score < best_score) {
        best_score = score;
        bestS = fitk.A;
        best_rms = fitk.rms;
        best_center = fitk.b;
      }
    }

    if (nb < 1) return false;

    // Fit bias(T) from bin centers
    TempBias3<T> biasT;
    if (nb >= 2) {
      // pack into fixed array form for template
      // (K_TBINS is small; we can just do a tiny copy)
      Eigen::Matrix<T,3,1> btmp[K_TBINS];
      T ttmp[K_TBINS];
      for (int i = 0; i < nb; ++i) { btmp[i] = centers[i]; ttmp[i] = temps[i]; }
      biasT = fit_temp_bias3<T, K_TBINS>(btmp, ttmp, nb, T0);
    } else {
      biasT.ok = true;
      biasT.T0 = T0;
      biasT.b0 = best_center;
      biasT.k.setZero();
    }

    out.ok = true;
    out.g = g;
    out.S = bestS;
    out.biasT = biasT;
    out.rms_mag = best_rms;
    return true;
  }
};

// ============================
// Calibrator: Magnetometer
// Strategy:
//  - Collect ~400 mag samples across orientations (rotate in 3D).
//  - Robust ellipsoid->unit sphere gives (b, A_unit).
//  - Preserve µT magnitude by scaling output so that median(||m_raw-b||) is maintained.
//    This is practical when your magnetometer already reports µT scale reasonably.
// ============================
template <typename T, int N>
struct MagCalibrator {
  using Vec3 = Eigen::Matrix<T,3,1>;
  SampleBuffer3<T, N> buf;

  // sanity gates
  T min_norm_uT = T(5);
  T max_norm_uT = T(200);

  void clear() { buf.clear(); }

  bool addSample(const Vec3& m_raw_uT) {
    if (!isfinite3(m_raw_uT)) return false;
    const T nrm = m_raw_uT.norm();
    if (nrm < min_norm_uT || nrm > max_norm_uT) return false; // likely bad read/saturation
    return buf.push(m_raw_uT, T(0));
  }

  bool fit(MagCalibration<T>& out, int robust_iters = 3, T trim_frac = T(0.15)) const {
    if (buf.n < 80) return false;

    auto fit0 = ellipsoid_to_sphere_robust<T>(buf.v, buf.n, T(1), robust_iters, trim_frac);
    if (!fit0.ok) return false;

    // Estimate field magnitude in µT from fitted center (robust median radius)
    T radii[400];
    int m = 0;
    for (int i = 0; i < buf.n; ++i) {
      radii[m++] = (buf.v[i] - fit0.b).norm();
    }
    T B_med = median_of_array(radii, m);
    // Typical Earth field is ~25-65 uT; allow broad
    if (B_med < T(12) || B_med > T(120)) return false;

    // Scale to preserve µT magnitude
    out.ok = true;
    out.b = fit0.b;
    out.A = fit0.A * B_med; // fit0.A maps to unit sphere; multiply by B_med -> µT-like output
    out.field_uT = B_med;
    out.rms = fit0.rms * B_med;
    return true;
  }
};

// ============================
// Calibrator: Gyroscope bias(T)
// Strategy:
//  - Collect stationary gyro samples (w_raw) while not moving.
//  - Bin by temperature and estimate mean bias per bin.
//  - Fit bias(T) via regression.
// ============================
template <typename T, int N, int K_TBINS = 8>
struct GyroCalibrator {
  using Vec3 = Eigen::Matrix<T,3,1>;
  SampleBuffer3<T, N> buf;

  T T0 = T(25);

  // stationary gate thresholds
  T max_gyro_norm = T(0.08);   // rad/s
  T max_accel_dev = T(1.0);    // m/s^2 (requires caller to supply accel too)
  T g = T(9.80665);

  void clear() { buf.clear(); }

  bool addSample(const Vec3& w_raw, const Vec3& a_raw, T tempC) {
    if (!isfinite3(w_raw) || !isfinite3(a_raw) || !finitef((float)tempC)) return false;
    if (w_raw.norm() > max_gyro_norm) return false;
    if ((T)fabs((double)(a_raw.norm() - g)) > max_accel_dev) return false;
    return buf.push(w_raw, tempC);
  }

  bool fit(GyroCalibration<T>& out) const {
    if (buf.n < 80) return false;

    // temp bins
    T tmin = buf.tempC[0], tmax = buf.tempC[0];
    for (int i = 1; i < buf.n; ++i) { tmin = (buf.tempC[i] < tmin ? buf.tempC[i] : tmin); tmax = (buf.tempC[i] > tmax ? buf.tempC[i] : tmax); }
    const T trange = tmax - tmin;
    const T binW = (trange > T(1e-3)) ? (trange / (T)K_TBINS) : T(1);

    Eigen::Matrix<T,3,1> meanB[K_TBINS];
    T temps[K_TBINS];
    int cnt[K_TBINS];
    for (int k = 0; k < K_TBINS; ++k) { meanB[k].setZero(); cnt[k]=0; temps[k]=tmin + (T(k)+T(0.5))*binW; }

    for (int i = 0; i < buf.n; ++i) {
      int k = 0;
      if (trange > T(1e-3)) {
        k = (int)floor((double)((buf.tempC[i] - tmin) / binW));
        k = clamp<int>(k, 0, K_TBINS - 1);
      }
      meanB[k] += buf.v[i];
      cnt[k] += 1;
    }

    Eigen::Matrix<T,3,1> bcenters[K_TBINS];
    T tcenters[K_TBINS];
    int nb = 0;
    for (int k = 0; k < K_TBINS; ++k) {
      if (cnt[k] < 20) continue;
      bcenters[nb] = meanB[k] / (T)cnt[k];
      tcenters[nb] = temps[k];
      nb++;
    }
    if (nb < 1) return false;

    TempBias3<T> biasT;
    if (nb >= 2) {
      biasT = fit_temp_bias3<T, K_TBINS>(bcenters, tcenters, nb, T0);
    } else {
      biasT.ok = true;
      biasT.T0 = T0;
      biasT.b0 = bcenters[0];
      biasT.k.setZero();
    }

    out.ok = true;
    out.S = Eigen::Matrix<T,3,3>::Identity();
    out.biasT = biasT;
    return true;
  }
};

} // namespace imu_cal
