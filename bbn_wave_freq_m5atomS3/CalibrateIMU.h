#pragma once

/*
  Copyright 2026, Mikhail Grushinskiy

  Modern embedded-friendly IMU calibration for Arduino + Eigen.
  Units:
    accel: m/s^2
    mag:   uT
    gyro:  rad/s

  NOTE:
  - Keep away from metal during MAG (boat rails, desk legs, speakers, USB cables, etc).

*/

#include <stdint.h>
#include <cmath>

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#else
#include <ArduinoEigenDense.h>
#endif

namespace imu_cal {

// Config / limits
static constexpr int IMU_CAL_MAX_SAMPLES = 400;

// Failure reasons
enum class FitFail : uint8_t {
  OK = 0,

  // Generic / inputs
  BAD_ARG,
  TOO_FEW_SAMPLES,
  NON_FINITE_INPUT,

  // Coverage / geometry
  DEGENERATE_COVERAGE,

  // Linear algebra / model reconstruction
  SOLVE_LDLT_FAIL,
  MODEL_LU_NOT_INVERTIBLE,
  MODEL_S_NONPOSITIVE,
  MODEL_SPD_PROJECT_FAIL,
  MODEL_CHOLESKY_FAIL,

  // Robust loop / scoring
  INLIERS_TOO_FEW,

  // Domain-specific
  MAG_FIELD_OUT_OF_RANGE,
  TEMP_BINS_EMPTY,

  // Accel-specific: fitted S would rotate axes / be unphysical
  ACCEL_S_UNPHYSICAL,
};

static inline const char* fitFailStr(FitFail f) {
  switch (f) {
    case FitFail::OK: return "OK";
    case FitFail::BAD_ARG: return "BAD_ARG";
    case FitFail::TOO_FEW_SAMPLES: return "TOO_FEW_SAMPLES";
    case FitFail::NON_FINITE_INPUT: return "NON_FINITE_INPUT";
    case FitFail::DEGENERATE_COVERAGE: return "DEGENERATE_COVERAGE";
    case FitFail::SOLVE_LDLT_FAIL: return "SOLVE_LDLT_FAIL";
    case FitFail::MODEL_LU_NOT_INVERTIBLE: return "MODEL_LU_NOT_INVERTIBLE";
    case FitFail::MODEL_S_NONPOSITIVE: return "MODEL_S_NONPOSITIVE";
    case FitFail::MODEL_SPD_PROJECT_FAIL: return "MODEL_SPD_PROJECT_FAIL";
    case FitFail::MODEL_CHOLESKY_FAIL: return "MODEL_CHOLESKY_FAIL";
    case FitFail::INLIERS_TOO_FEW: return "INLIERS_TOO_FEW";
    case FitFail::MAG_FIELD_OUT_OF_RANGE: return "MAG_FIELD_OUT_OF_RANGE";
    case FitFail::TEMP_BINS_EMPTY: return "TEMP_BINS_EMPTY";
    case FitFail::ACCEL_S_UNPHYSICAL: return "ACCEL_S_UNPHYSICAL";
    default: return "UNKNOWN";
  }
}

// Utilities
template <typename T>
static inline T clamp(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }

template <typename T>
static inline bool finiteT(T x) { return std::isfinite((double)x); }

template <typename T>
static inline bool isfinite3(const Eigen::Matrix<T,3,1>& v) {
  return finiteT(v.x()) && finiteT(v.y()) && finiteT(v.z());
}

// SPD projection for 3x3 symmetric matrices
// Clamps eigenvalues to enforce positive-definiteness.
// Returns false if eigendecomposition fails or result is non-finite.
template <typename T>
static inline bool project_spd_3x3(
    Eigen::Matrix<T,3,3>& M,
    T eps_rel = T(1e-5),
    T eps_abs = T(1e-7))
{
  using Mat3 = Eigen::Matrix<T,3,3>;
  // Symmetrize first
  M = T(0.5) * (M + M.transpose());

  Eigen::SelfAdjointEigenSolver<Mat3> es(M);
  if (es.info() != Eigen::Success) return false;

  Eigen::Matrix<T,3,1> eval = es.eigenvalues();
  Mat3 evec = es.eigenvectors();
  if (!(eval.array().isFinite().all()) || !(evec.array().isFinite().all())) return false;

  const T max_e = eval.maxCoeff();
  const T floor_e = (max_e > T(0)) ? (eps_rel * max_e) : eps_abs;
  const T min_e = (floor_e > eps_abs) ? floor_e : eps_abs;

  eval.x() = (eval.x() < min_e ? min_e : eval.x());
  eval.y() = (eval.y() < min_e ? min_e : eval.y());
  eval.z() = (eval.z() < min_e ? min_e : eval.z());

  M = evec * eval.asDiagonal() * evec.transpose();
  M = T(0.5) * (M + M.transpose());
  return (M.array().isFinite().all());
}

template <typename T>
static inline bool degeneracy_check_coverage3(
    const Eigen::Matrix<T,3,1>* x, int n,
    T expected_radius,                 // e.g. g for accel, ~B for mag
    T min_axis_span_mult = T(0.90),    // require span >= 0.90*R on each axis (after centering)
    T min_sign_frac      = T(0.08),    // require >=8% of samples on both +/- side per axis (after centering)
    T min_cov_det        = T(1e-3))    // covariance determinant threshold for unit directions
{
  if (!x || n < 12) return false;
  if (!(expected_radius > T(0))) return false;

  using Vec3 = Eigen::Matrix<T,3,1>;
  using Mat3 = Eigen::Matrix<T,3,3>;

  // Mean-center to remove bias offsets.
  Vec3 mu = Vec3::Zero();
  for (int i = 0; i < n; ++i) mu += x[i];
  mu *= (T(1) / (T)n);

  // Axis span and sign balance checks (centered).
  Vec3 mn = Vec3::Constant( T(1e30));
  Vec3 mx = Vec3::Constant(-T(1e30));
  int pos[3] = {0,0,0};
  int neg[3] = {0,0,0};

  // Direction covariance on unit vectors u = (x-mu)/||x-mu||
  Mat3 C = Mat3::Zero();
  int m = 0;

  int valid = 0; // count of finite centered samples used for sign/span tests

  for (int i = 0; i < n; ++i) {
    Vec3 d = x[i] - mu;
    if (!isfinite3(d)) continue;

    ++valid;

    // track min/max
    mn.x() = (d.x() < mn.x() ? d.x() : mn.x());
    mn.y() = (d.y() < mn.y() ? d.y() : mn.y());
    mn.z() = (d.z() < mn.z() ? d.z() : mn.z());
    mx.x() = (d.x() > mx.x() ? d.x() : mx.x());
    mx.y() = (d.y() > mx.y() ? d.y() : mx.y());
    mx.z() = (d.z() > mx.z() ? d.z() : mx.z());

    // sign counts
    if (d.x() > T(0)) pos[0]++; else if (d.x() < T(0)) neg[0]++;
    if (d.y() > T(0)) pos[1]++; else if (d.y() < T(0)) neg[1]++;
    if (d.z() > T(0)) pos[2]++; else if (d.z() < T(0)) neg[2]++;

    const T dn = d.norm();
    if (!(dn > T(1e-9))) continue;

    Vec3 u = d / dn;
    if (!isfinite3(u)) continue;

    C.noalias() += u * u.transpose();
    ++m;
  }

  if (valid < 12) return false;
  if (m < 12) return false;

  // Axis span must be large enough (requires coverage in each direction).
  const Vec3 span = mx - mn;
  const T min_span = min_axis_span_mult * expected_radius;

  if (!(span.x() > min_span && span.y() > min_span && span.z() > min_span)) return false;

  // Must have both positive and negative samples on each axis (after centering).
  const int min_side = (int)ceil((double)valid * (double)min_sign_frac);
  for (int a = 0; a < 3; ++a) {
    if (pos[a] < min_side || neg[a] < min_side) return false;
  }

  // Normalize covariance; for uniform sphere C ~ (1/3)I, det ~ 1/27 ≈ 0.037.
  C *= (T(1) / (T)m);

  // Determinant collapses to ~0 if samples are planar/linear in direction space.
  const T detC = C.determinant();
  if (!finiteT(detC) || detC < min_cov_det) return false;

  return true;
}

template <typename T>
static inline T sqr(T x) { return x*x; }

template <typename T>
static inline void sort_small(T* a, int n) {
  // Simple insertion sort (n <= ~400)
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

// helpers to prevent accel axis rotation
// Off-diagonal RMS for a 3x3 matrix (6 off-diagonal entries)
template <typename T>
static inline T offdiag_rms_3x3(const Eigen::Matrix<T,3,3>& M) {
  const T s2 =
      M(0,1)*M(0,1) + M(0,2)*M(0,2) +
      M(1,0)*M(1,0) + M(1,2)*M(1,2) +
      M(2,0)*M(2,0) + M(2,1)*M(2,1);
  return (T)std::sqrt((double)(s2 / T(6)));
}

// SPD condition number using eigenvalues (assumes SPD or nearly SPD)
template <typename T>
static inline bool cond_spd_3x3(const Eigen::Matrix<T,3,3>& M, T& cond_out) {
  using Mat3 = Eigen::Matrix<T,3,3>;
  Mat3 S = T(0.5) * (M + M.transpose());
  Eigen::SelfAdjointEigenSolver<Mat3> es(S);
  if (es.info() != Eigen::Success) return false;
  auto ev = es.eigenvalues();
  if (!(ev.array().isFinite().all())) return false;
  const T emax = ev.maxCoeff();
  const T emin = ev.minCoeff();
  if (!(emin > T(0))) return false;
  cond_out = emax / emin;
  return finiteT(cond_out);
}

// Polar SPD factor: P = sqrt(A^T A). Removes any rotation from A.
template <typename T>
static inline bool polar_spd_factor_3x3(const Eigen::Matrix<T,3,3>& A, Eigen::Matrix<T,3,3>& P_out) {
  using Mat3 = Eigen::Matrix<T,3,3>;
  Mat3 G = A.transpose() * A;
  G = T(0.5) * (G + G.transpose());

  if (!project_spd_3x3<T>(G, T(1e-5), T(1e-7))) return false;

  Eigen::SelfAdjointEigenSolver<Mat3> es(G);
  if (es.info() != Eigen::Success) return false;

  auto V = es.eigenvectors();
  auto d = es.eigenvalues();
  if (!(V.array().isFinite().all()) || !(d.array().isFinite().all())) return false;

  d.x() = (T)std::sqrt((double)d.x());
  d.y() = (T)std::sqrt((double)d.y());
  d.z() = (T)std::sqrt((double)d.z());

  Mat3 P = V * d.asDiagonal() * V.transpose();
  P = T(0.5) * (P + P.transpose());
  if (!(P.array().isFinite().all())) return false;

  P_out = P;
  return true;
}

template <typename T>
static inline Eigen::Matrix<T,3,3> diag_only_from(const Eigen::Matrix<T,3,3>& M) {
  Eigen::Matrix<T,3,3> D = Eigen::Matrix<T,3,3>::Zero();
  D(0,0) = M(0,0);
  D(1,1) = M(1,1);
  D(2,2) = M(2,2);
  return D;
}

// Fixed-capacity sample buffer
template <typename T, int N>
struct SampleBuffer3 {
  static_assert(N > 0, "N must be positive");
  static_assert(N <= IMU_CAL_MAX_SAMPLES, "N exceeds IMU_CAL_MAX_SAMPLES (400)");

  using Vec3 = Eigen::Matrix<T,3,1>;

  Vec3 v[N];
  T    tempC[N];
  int  n = 0;

  void clear() { n = 0; }

  bool push(const Vec3& x, T tC = T(0)) {
    if (n >= N) return false;
    if (!isfinite3(x) || !finiteT(tC)) return false;
    v[n] = x;
    tempC[n] = tC;
    ++n;
    return true;
  }
};

// Ellipsoid -> sphere robust fit
template <typename T>
struct EllipsoidSphereFit {
  bool ok = false;
  FitFail reason = FitFail::BAD_ARG;

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
static FitFail solve_ellipsoid_params_trimmed_ex(
    const Eigen::Matrix<T,3,1>* x, int n,
    const bool* inlier,
    Eigen::Matrix<T,10,1>& p_out,
    T ridge_rel)
{
  using Mat10 = Eigen::Matrix<T,10,10>;
  using Vec10 = Eigen::Matrix<T,10,1>;

  if (!x || n <= 0) return FitFail::BAD_ARG;

  Mat10 H = Mat10::Zero();
  Vec10 g = Vec10::Zero();

  int used = 0;
  for (int i = 0; i < n; ++i) {
    if (inlier && !inlier[i]) continue;
    if (!isfinite3<T>(x[i])) continue;

    Vec10 d;
    build_row_d(x[i], d);
    if (!(d.array().isFinite().all())) continue;

    H.noalias() += d * d.transpose();
    g.noalias() += d; // D^T * 1
    ++used;
  }
  if (used < 10) return FitFail::TOO_FEW_SAMPLES;

  // Scale ridge by feature energy for better invariance
  const T tr = H.trace() / T(10);
  const T ridge = ridge_rel * (tr + T(1e-12));
  H.diagonal().array() += ridge;

  Eigen::LDLT<Mat10> ldlt(H);
  if (ldlt.info() != Eigen::Success) return FitFail::SOLVE_LDLT_FAIL;

  Vec10 p = ldlt.solve(g);
  if (!(p.array().isFinite().all())) return FitFail::NON_FINITE_INPUT;

  p_out = p;
  return FitFail::OK;
}

template <typename T>
static EllipsoidSphereFit<T> ellipsoid_to_sphere_robust(
    const Eigen::Matrix<T,3,1>* x, int n,
    T R_target,
    int robust_iters = 3,
    T trim_frac = T(0.15),     // drop worst ~15% by |e| each iteration
    T ridge_rel = T(1e-6),
    T expected_radius_for_checks = T(0)) // optional early coverage check
{
  EllipsoidSphereFit<T> out;
  out.ok = false;
  out.reason = FitFail::BAD_ARG;

  if (!x || n < 12) { out.reason = FitFail::TOO_FEW_SAMPLES; return out; }
  if (n > IMU_CAL_MAX_SAMPLES) { out.reason = FitFail::BAD_ARG; return out; }
  if (!(R_target > T(0))) { out.reason = FitFail::BAD_ARG; return out; }

  // Optional early degeneracy check (coverage/planarity)
  if (expected_radius_for_checks > T(0)) {
    if (!degeneracy_check_coverage3<T>(
          x, n,
          expected_radius_for_checks,
          T(0.90),
          T(0.08),
          T(1e-3)))
    {
      out.reason = FitFail::DEGENERATE_COVERAGE;
      return out;
    }
  }

  bool inlier[IMU_CAL_MAX_SAMPLES];
  for (int i = 0; i < n; ++i) inlier[i] = true;

  // Working state
  Eigen::Matrix<T,10,1> p;
  Eigen::Matrix<T,3,1>  b = Eigen::Matrix<T,3,1>::Zero();
  Eigen::Matrix<T,3,3>  A_unit = Eigen::Matrix<T,3,3>::Identity();

  // Robust params
  robust_iters = (robust_iters <= 0 ? 1 : robust_iters);
  trim_frac = clamp<T>(trim_frac, T(0), T(0.45));

  // Residual buffers (unit-sphere space): e = r - 1
  T e[IMU_CAL_MAX_SAMPLES];
  T abs_e[IMU_CAL_MAX_SAMPLES];
  T tmp[IMU_CAL_MAX_SAMPLES];

  // Helper: rebuild b/A_unit from p and validate
  auto build_model_from_p = [&](const Eigen::Matrix<T,10,1>& p_in) -> FitFail {
    Eigen::Matrix<T,3,3> Q;
    Q << p_in(0), p_in(3), p_in(4),
         p_in(3), p_in(1), p_in(5),
         p_in(4), p_in(5), p_in(2);
    Eigen::Matrix<T,3,1> q;
    q << p_in(6), p_in(7), p_in(8);
    const T c = p_in(9);

    if (!(Q.array().isFinite().all()) || !(q.array().isFinite().all()) || !finiteT(c)) {
      return FitFail::NON_FINITE_INPUT;
    }

    // b = -Q^{-1} q
    Eigen::FullPivLU<Eigen::Matrix<T,3,3>> lu(Q);
    if (!lu.isInvertible()) return FitFail::MODEL_LU_NOT_INVERTIBLE;
    b = -lu.solve(q);
    if (!isfinite3(b)) return FitFail::NON_FINITE_INPUT;

    const double btQb =
        b.template cast<double>().dot(
            Q.template cast<double>() * b.template cast<double>());
    const double sD = 1.0 + btQb - (double)c;

    // Accept either sign; only reject near-zero (degenerate scaling / numerical collapse)
    if (!std::isfinite(sD) || fabs(sD) <= 1e-12) return FitFail::MODEL_S_NONPOSITIVE;

    // IMPORTANT: use signed s so that if Q is negative definite and s is negative,
    // M = Q/s is still positive definite.
    const T s = (T)sD;

    // M = Q / s (SIGNED), then symmetrize
    Eigen::Matrix<T,3,3> M = Q / s;
    M = T(0.5) * (M + M.transpose());

    // project to SPD to avoid borderline LLT failures
    if (!project_spd_3x3<T>(M, T(1e-5), T(1e-7))) return FitFail::MODEL_SPD_PROJECT_FAIL;

    // Cholesky: M = U^T U
    Eigen::LLT<Eigen::Matrix<T,3,3>> llt(M);
    if (llt.info() != Eigen::Success) return FitFail::MODEL_CHOLESKY_FAIL;
    A_unit = llt.matrixU();
    return FitFail::OK;
  };

  for (int it = 0; it < robust_iters; ++it) {
    {
      FitFail fr = solve_ellipsoid_params_trimmed_ex<T>(x, n, inlier, p, ridge_rel);
      if (fr != FitFail::OK) { out.reason = fr; return out; }
    }
    {
      FitFail fr = build_model_from_p(p);
      if (fr != FitFail::OK) { out.reason = fr; return out; }
    }

    // Compute residuals for ALL points: e = ||A_unit(x-b)|| - 1
    for (int i = 0; i < n; ++i) {
      const T r = (A_unit * (x[i] - b)).norm();
      T ei = r - T(1);
      if (!finiteT(ei)) {
        ei = T(1e9);
      }
      e[i] = ei;
      abs_e[i] = (T)fabs((double)ei);
    }

    // Robust scale via MAD
    for (int i = 0; i < n; ++i) tmp[i] = e[i];
    const T mad = robust_mad(tmp, n);

    const T mad_floor = T(1e-4);
    const T mad_eff   = (mad > mad_floor ? mad : mad_floor);
    const T thr_mad = T(5.2) * mad_eff + T(1e-6);

    // Quantile trim threshold (drop worst trim_frac)
    T thr_trim = thr_mad;
    if (trim_frac > T(0)) {
      for (int i = 0; i < n; ++i) tmp[i] = abs_e[i];
      sort_small(tmp, n);
      int keep = (int)floor((double)((T(1) - trim_frac) * (T)n));
      keep = clamp<int>(keep, 10, n);
      thr_trim = tmp[keep - 1];
    }

    // Combine thresholds
    T thr = (thr_trim < thr_mad ? thr_trim : thr_mad);

    // Update inliers
    int used = 0;
    for (int i = 0; i < n; ++i) {
      inlier[i] = (abs_e[i] <= thr);
      if (inlier[i]) ++used;
    }

    // If too aggressive, relax to looser threshold
    if (used < 10) {
      thr = (thr_trim > thr_mad ? thr_trim : thr_mad);
      used = 0;
      for (int i = 0; i < n; ++i) {
        inlier[i] = (abs_e[i] <= thr);
        if (inlier[i]) ++used;
      }
      if (used < 10) { out.reason = FitFail::INLIERS_TOO_FEW; return out; }
    }
  }

  // Final solve using FINAL inlier mask
  {
    FitFail fr = solve_ellipsoid_params_trimmed_ex<T>(x, n, inlier, p, ridge_rel);
    if (fr != FitFail::OK) { out.reason = fr; return out; }
  }
  {
    FitFail fr = build_model_from_p(p);
    if (fr != FitFail::OK) { out.reason = fr; return out; }
  }

  // Final scale A to target radius
  const Eigen::Matrix<T,3,3> A = A_unit * R_target;

  int used = 0;
  T sum_e2 = 0;
  T radii[IMU_CAL_MAX_SAMPLES];

  for (int i = 0; i < n; ++i) {
    if (!inlier[i]) continue;
    const T r = (A * (x[i] - b)).norm();
    if (!finiteT(r)) continue;
    radii[used] = r;
    const T er = r - R_target;
    sum_e2 += er * er;
    ++used;
  }
  if (used < 10) { out.reason = FitFail::INLIERS_TOO_FEW; return out; }

  const T medr = median_of_array(radii, used);

  out.ok = true;
  out.reason = FitFail::OK;
  out.b = b;
  out.A = A;
  out.rms = (T)sqrt((double)(sum_e2 / (T)used));
  out.median_r = medr;
  out.used = used;
  return out;
}

// Temperature model: bias(T) = b0 + k*(T - T0)
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

// Calibration outputs
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

// Accel post-fit gravity renormalization
//
// Goal: after fitting S and bias(T), enforce mean ||S*(a_raw - bias(T))|| == g
// over gated (near-static) samples. This scales S by a scalar only.
template <typename T>
static inline Eigen::Matrix<T,3,1> bias_at_temp_(const TempBias3<T>& bt, T tempC) {
  return bt.bias(tempC);
}

template <typename T, typename AccelBufT>
static inline bool post_scale_accel_S_to_match_g_(
    const AccelBufT& buf,
    AccelCalibration<T>& out,
    T norm_gate_lo = T(0.90),    // gross sanity gate only (still used)
    T norm_gate_hi = T(1.10),
    T scale_clamp_lo = T(0.85),  // safety clamp on the scale factor
    T scale_clamp_hi = T(1.15))
{
  if (!out.ok) return false;
  if (!(out.g > T(0))) return false;

  const int n = (int)buf.n;
  if (n < 10) return false;

  const T g  = out.g;
  const T lo = norm_gate_lo * g;
  const T hi = norm_gate_hi * g;

  // Collect norms in current calibrated space (pre-scaling)
  T norms[IMU_CAL_MAX_SAMPLES];
  int m = 0;

  for (int i = 0; i < n; ++i) {
    const Eigen::Matrix<T,3,1> a_raw = buf.v[i];
    const T tempC = (T)buf.tempC[i];

    if (!isfinite3(a_raw)) continue;
    if (!finiteT(tempC)) continue;

    const Eigen::Matrix<T,3,1> b = bias_at_temp_(out.biasT, tempC);
    const Eigen::Matrix<T,3,1> a_cal = out.S * (a_raw - b);
    const T an = a_cal.norm();
    if (!finiteT(an) || !(an > T(0))) continue;

    // Optional gross gate (caller may widen on fallback)
    if (an < lo || an > hi) continue;

    norms[m++] = an;
    if (m >= IMU_CAL_MAX_SAMPLES) break;
  }

  if (m < 10) return false;

  // Robust center (median)
  T tmp[IMU_CAL_MAX_SAMPLES];
  for (int i = 0; i < m; ++i) tmp[i] = norms[i];
  const T med = median_of_array(tmp, m);
  if (!finiteT(med) || !(med > T(0))) return false;

  // Robust scale (MAD about median); robust_mad() overwrites its input
  for (int i = 0; i < m; ++i) tmp[i] = norms[i];
  T mad = robust_mad(tmp, m);

  const T mad_floor = T(1e-4) * g;          // scale-aware floor
  const T mad_eff   = (mad > mad_floor ? mad : mad_floor);
  const T thr       = T(4.0) * mad_eff + T(1e-6);

  // Inliers around the median (reject motion / transitions)
  T inl[IMU_CAL_MAX_SAMPLES];
  int mm = 0;
  for (int i = 0; i < m; ++i) {
    const T di = (T)fabs((double)(norms[i] - med));
    if (di <= thr) inl[mm++] = norms[i];
  }

  // If too aggressive, fall back to using all gated samples
  if (mm < 10) {
    for (int i = 0; i < m; ++i) inl[i] = norms[i];
    mm = m;
  }

  // Typical gravity magnitude from inliers = median
  for (int i = 0; i < mm; ++i) tmp[i] = inl[i];
  const T med2 = median_of_array(tmp, mm);
  if (!finiteT(med2) || !(med2 > T(0))) return false;

  // Final scale
  T scale = g / med2;
  if (!finiteT(scale)) return false;

  if (scale < scale_clamp_lo) scale = scale_clamp_lo;
  if (scale > scale_clamp_hi) scale = scale_clamp_hi;

  // Apply scalar renorm
  out.S *= scale;

  // Update rms_mag over the same inlier set (no need to recompute vectors;
  // scaling S by scalar scales norms linearly)
  {
    T sse = T(0);
    for (int i = 0; i < mm; ++i) {
      const T an2 = inl[i] * scale;
      const T e   = an2 - g;
      sse += e * e;
    }
    out.rms_mag = (mm > 0) ? (T)std::sqrt((double)(sse / (T)mm)) : T(0);
  }

  return true;
}

// Calibrator: Accel
template <typename T, int N, int K_TBINS = 8>
struct AccelCalibrator {
  static_assert(N <= IMU_CAL_MAX_SAMPLES, "N exceeds IMU_CAL_MAX_SAMPLES (400)");
  static_assert(K_TBINS > 0 && K_TBINS <= 16, "K_TBINS unreasonable");

  using Vec3 = Eigen::Matrix<T,3,1>;
  SampleBuffer3<T, N> buf;

  T g = T(9.80665);
  T T0 = T(25);

  // acceptance gates
  T max_gyro_for_static = T(0.07);   // rad/s
  T accel_mag_tol = T(0.25);         // m/s^2, accept if | |a| - g | < tol

  // prevent axis rotation
  enum class AccelSMode : uint8_t { PolarSPD = 0, DiagonalOnly = 1 };

  // Recommended default for compass/attitude: DiagonalOnly
  AccelSMode accel_S_mode = AccelSMode::DiagonalOnly;  // AccelSMode::PolarSPD; 

  // Plausibility gates (dimensionless)
  T accel_diag_lo = T(0.80);
  T accel_diag_hi = T(1.25);
  T accel_max_offdiag_rms = T(0.05); // only used in PolarSPD mode
  T accel_max_cond = T(3.5);

  // Workspace to reduce stack use in fit()
  struct Workspace {
    uint16_t idx[K_TBINS][N];
    int      nbin[K_TBINS];
    T        tsum[K_TBINS];
    Vec3     xscratch[N];
    Vec3     centers[K_TBINS];
    T        temps[K_TBINS];
  };
  mutable Workspace ws_;

  // last failure
  mutable FitFail last_fail_ = FitFail::OK;
  FitFail lastFail() const { return last_fail_; }

  void clear() { buf.clear(); }

  bool addSample(const Vec3& a_raw, const Vec3& w_raw, T tempC) {
    if (!isfinite3(a_raw) || !isfinite3(w_raw) || !finiteT(tempC)) return false;
    const T amag = a_raw.norm();
    const T wmag = w_raw.norm();
    if ((T)fabs((double)(amag - g)) > accel_mag_tol) return false;
    if (wmag > max_gyro_for_static) return false;
    return buf.push(a_raw, tempC);
  }

  // optional reason out
  bool fit(AccelCalibration<T>& out, int robust_iters = 3, T trim_frac = T(0.15), FitFail* reason_out = nullptr) const {
    last_fail_ = FitFail::OK;
    out.ok = false;

    if (buf.n < 80) { last_fail_ = FitFail::TOO_FEW_SAMPLES; if (reason_out) *reason_out = last_fail_; return false; }

    // Temperature range
    T tmin = buf.tempC[0], tmax = buf.tempC[0];
    for (int i = 1; i < buf.n; ++i) {
      tmin = (buf.tempC[i] < tmin ? buf.tempC[i] : tmin);
      tmax = (buf.tempC[i] > tmax ? buf.tempC[i] : tmax);
    }
    const T trange = tmax - tmin;
    const T binW = (trange > T(1e-3)) ? (trange / (T)K_TBINS) : T(1);

    auto& idx   = ws_.idx;
    auto& nbin  = ws_.nbin;
    auto& tsum  = ws_.tsum;
    auto& xs    = ws_.xscratch;
    auto& centers = ws_.centers;
    auto& temps   = ws_.temps;

    for (int k = 0; k < K_TBINS; ++k) { nbin[k] = 0; tsum[k] = T(0); }

    for (int i = 0; i < buf.n; ++i) {
      int k = 0;
      if (trange > T(1e-3)) {
        k = (int)floor((double)((buf.tempC[i] - tmin) / binW));
        k = clamp<int>(k, 0, K_TBINS - 1);
      }
      int& nk = nbin[k];
      if (nk < N) {
        idx[k][nk] = (uint16_t)i;
        tsum[k] += buf.tempC[i];
        ++nk;
      }
    }

    int nb = 0;

    // Choose S from best bin
    T best_score = T(1e30);
    Eigen::Matrix<T,3,3> bestS = Eigen::Matrix<T,3,3>::Identity();
    T best_rms = T(0);
    Vec3 best_center = Vec3::Zero();

    FitFail worst_bin_reason = FitFail::TEMP_BINS_EMPTY;

    for (int k = 0; k < K_TBINS; ++k) {
      if (nbin[k] < 30) continue;

      for (int j = 0; j < nbin[k]; ++j) {
        xs[j] = buf.v[(int)idx[k][j]];
      }
      const T tmean = tsum[k] / (T)nbin[k];

      auto fitk = ellipsoid_to_sphere_robust<T>(
        xs, nbin[k], g,
        robust_iters, trim_frac,
        T(1e-6),
        g);

      if (!fitk.ok) { worst_bin_reason = fitk.reason; continue; }

      centers[nb] = fitk.b;
      temps[nb]   = tmean;
      nb++;

      // convert fit matrix to axis-safe S
      // 1) Strip rotation using polar SPD factor
      Eigen::Matrix<T,3,3> S_spd;
      if (!polar_spd_factor_3x3<T>(fitk.A, S_spd)) {
        worst_bin_reason = FitFail::MODEL_SPD_PROJECT_FAIL;
        continue;
      }

      // 2) Optionally force diagonal-only
      Eigen::Matrix<T,3,3> S_use = (accel_S_mode == AccelSMode::DiagonalOnly)
                                   ? diag_only_from<T>(S_spd)
                                   : S_spd;

      // 3) Plausibility gates
      const T d0 = S_use(0,0), d1 = S_use(1,1), d2 = S_use(2,2);
      if (!(d0 >= accel_diag_lo && d0 <= accel_diag_hi &&
            d1 >= accel_diag_lo && d1 <= accel_diag_hi &&
            d2 >= accel_diag_lo && d2 <= accel_diag_hi))
      {
        worst_bin_reason = FitFail::ACCEL_S_UNPHYSICAL;
        continue;
      }

      T condv = T(0);
      if (!cond_spd_3x3<T>(S_spd, condv) || condv > accel_max_cond) {
        worst_bin_reason = FitFail::ACCEL_S_UNPHYSICAL;
        continue;
      }

      if (accel_S_mode != AccelSMode::DiagonalOnly) {
        const T offr = offdiag_rms_3x3<T>(S_use);
        if (offr > accel_max_offdiag_rms) {
          worst_bin_reason = FitFail::ACCEL_S_UNPHYSICAL;
          continue;
        }
      }

      // score: rms + small penalty for fewer used points + tiny penalty for cond/offdiag
      T score = fitk.rms + T(0.2) * (T(50) / (T)fitk.used);
      score += T(0.02) * (condv - T(1));
      if (accel_S_mode != AccelSMode::DiagonalOnly) {
        score += T(0.15) * offdiag_rms_3x3<T>(S_use);
      }

      if (score < best_score) {
        best_score = score;
        bestS = S_use;           // axis-safe S stored here
        best_rms = fitk.rms;
        best_center = fitk.b;
      }
    }

    if (nb < 1) {
      last_fail_ = worst_bin_reason;
      if (last_fail_ == FitFail::OK) last_fail_ = FitFail::TEMP_BINS_EMPTY;
      if (reason_out) *reason_out = last_fail_;
      return false;
    }

    // Fit bias(T) from bin centers
    TempBias3<T> biasT;
    if (nb >= 2) {
      Eigen::Matrix<T,3,1> btmp[K_TBINS];
      T ttmp[K_TBINS];
      for (int i = 0; i < nb; ++i) { btmp[i] = centers[i]; ttmp[i] = temps[i]; }
      biasT = fit_temp_bias3<T, K_TBINS>(btmp, ttmp, nb, T0);
      // If regression ill-conditioned, we still succeed but keep k=0 fallback:
      if (!biasT.ok) {
        biasT.ok = true;
        biasT.T0 = T0;
        biasT.b0 = best_center;
        biasT.k.setZero();
      }
    } else {
      biasT.ok = true;
      biasT.T0 = T0;
      biasT.b0 = best_center;
      biasT.k.setZero();
    }

    out.ok = true;
    out.g = g;
    out.S = bestS;          // axis-safe S used here
    out.biasT = biasT;
    out.rms_mag = best_rms;

    // Tight-first g renorm. Relax only if it can't find enough near-static samples.
    bool did_scale = post_scale_accel_S_to_match_g_<T>(this->buf, out,
                                                       T(0.985), T(1.015),  // ±1.5%
                                                       T(0.97),  T(1.03));  // clamp scale
    if (!did_scale) {
      did_scale = post_scale_accel_S_to_match_g_<T>(this->buf, out,
                                                    T(0.95),  T(1.05),   // ±5%
                                                    T(0.93),  T(1.07));  // clamp scale
    }
    if (!did_scale) {
      // Last-chance fallback so real datasets still succeed.
      did_scale = post_scale_accel_S_to_match_g_<T>(this->buf, out,
                                                    T(0.90),  T(1.10),
                                                    T(0.90),  T(1.10));
    }
    
    if (!did_scale) {
      last_fail_ = FitFail::ACCEL_S_UNPHYSICAL;
      if (reason_out) *reason_out = last_fail_;
      return false;
    }
    
    last_fail_ = FitFail::OK;
    if (reason_out) *reason_out = FitFail::OK;
    return true;
  }
};

// Calibrator: Magnetometer
template <typename T, int N>
struct MagCalibrator {
  static_assert(N <= IMU_CAL_MAX_SAMPLES, "N exceeds IMU_CAL_MAX_SAMPLES (400)");
  using Vec3 = Eigen::Matrix<T,3,1>;
  SampleBuffer3<T, N> buf;
  mutable Vec3 xs_[IMU_CAL_MAX_SAMPLES]; // scaled samples workspace

  // sanity gates
  T min_norm_uT = T(5);
  T max_norm_uT = T(200);

  mutable FitFail last_fail_ = FitFail::OK;
  FitFail lastFail() const { return last_fail_; }

  void clear() { buf.clear(); }

  bool addSample(const Vec3& m_raw_uT) {
    if (!isfinite3(m_raw_uT)) return false;
    const T nrm = m_raw_uT.norm();
    if (nrm < min_norm_uT || nrm > max_norm_uT) return false;
    return buf.push(m_raw_uT, T(0));
  }

  bool fit(MagCalibration<T>& out,
           int robust_iters = 3,
           T trim_frac = T(0.15),
           T ridge_rel = T(1e-6),
           FitFail* reason_out = nullptr) const
  {
    last_fail_ = FitFail::OK;
    out.ok = false;

    if (buf.n < 80) {
      last_fail_ = FitFail::TOO_FEW_SAMPLES;
      if (reason_out) *reason_out = last_fail_;
      return false;
    }

    // Mean (for B_est)
    Vec3 mu = Vec3::Zero();
    for (int i = 0; i < buf.n; ++i) mu += buf.v[i];
    mu *= (T(1) / (T)buf.n);

    // Robust radius estimate in raw uT space
    T rad_mu[IMU_CAL_MAX_SAMPLES];
    for (int i = 0; i < buf.n; ++i) rad_mu[i] = (buf.v[i] - mu).norm();
    T B_est = median_of_array(rad_mu, buf.n);
    if (!(B_est > T(1e-6))) {
      last_fail_ = FitFail::DEGENERATE_COVERAGE;
      if (reason_out) *reason_out = last_fail_;
      return false;
    }

    // SCALE SAMPLES: x' = x / B_est (brings values ~O(1))
    for (int i = 0; i < buf.n; ++i) xs_[i] = buf.v[i] / B_est;

    // Fit in scaled space (expected radius ~1)
    auto fit0 = ellipsoid_to_sphere_robust<T>(
      xs_, buf.n, T(1),
      robust_iters, trim_frac,
      ridge_rel,
      T(1)   // expected_radius_for_checks in scaled space
    );

    if (!fit0.ok) {
      last_fail_ = fit0.reason;
      if (reason_out) *reason_out = last_fail_;
      return false;
    }

    // Unscale to uT space:
    // We want A_uT such that A_uT*(x_uT - b_uT) = fit0.A*(x_uT/B_est - fit0.b)
    // => b_uT = fit0.b * B_est, A_uT = fit0.A / B_est
    const Vec3 b_uT = fit0.b * B_est;
    const Eigen::Matrix<T,3,3> A_unit_uTinv = fit0.A / B_est;

    // Estimate field magnitude from median radius in raw uT space
    T radii[IMU_CAL_MAX_SAMPLES];
    int m = 0;
    for (int i = 0; i < buf.n; ++i) radii[m++] = (buf.v[i] - b_uT).norm();
    T B_med = median_of_array(radii, m);

    // Typical Earth field is ~25-65 uT; allow broad
    if (B_med < T(12) || B_med > T(120)) {
      last_fail_ = FitFail::MAG_FIELD_OUT_OF_RANGE;
      if (reason_out) *reason_out = last_fail_;
      return false;
    }

    out.ok = true;
    out.b = b_uT;
    out.A = A_unit_uTinv * B_med;  // calibrated output ~uT magnitude
    out.field_uT = B_med;
    out.rms = fit0.rms * B_med;

    last_fail_ = FitFail::OK;
    if (reason_out) *reason_out = FitFail::OK;
    return true;
  }
};

// Calibrator: Gyroscope bias(T)
template <typename T, int N, int K_TBINS = 8>
struct GyroCalibrator {
  static_assert(N <= IMU_CAL_MAX_SAMPLES, "N exceeds IMU_CAL_MAX_SAMPLES (400)");
  static_assert(K_TBINS > 0 && K_TBINS <= 16, "K_TBINS unreasonable");

  using Vec3 = Eigen::Matrix<T,3,1>;
  SampleBuffer3<T, N> buf;

  T T0 = T(25);

  // stationary gate thresholds
  T max_gyro_norm = T(0.07);    // rad/s
  T max_accel_dev = T(0.25);    // m/s^2
  T g = T(9.80665);

  mutable FitFail last_fail_ = FitFail::OK;
  FitFail lastFail() const { return last_fail_; }

  void clear() { buf.clear(); }

  bool addSample(const Vec3& w_raw, const Vec3& a_raw, T tempC) {
    if (!isfinite3(w_raw) || !isfinite3(a_raw) || !finiteT(tempC)) return false;
    if (w_raw.norm() > max_gyro_norm) return false;
    if ((T)fabs((double)(a_raw.norm() - g)) > max_accel_dev) return false;
    return buf.push(w_raw, tempC);
  }

  bool fit(GyroCalibration<T>& out, FitFail* reason_out = nullptr) const {
    last_fail_ = FitFail::OK;
    out.ok = false;

    if (buf.n < 80) { last_fail_ = FitFail::TOO_FEW_SAMPLES; if (reason_out) *reason_out = last_fail_; return false; }

    // Temperature range
    T tmin = buf.tempC[0], tmax = buf.tempC[0];
    for (int i = 1; i < buf.n; ++i) {
      tmin = (buf.tempC[i] < tmin ? buf.tempC[i] : tmin);
      tmax = (buf.tempC[i] > tmax ? buf.tempC[i] : tmax);
    }
    const T trange = tmax - tmin;
    const T binW = (trange > T(1e-3)) ? (trange / (T)K_TBINS) : T(1);

    Eigen::Matrix<T,3,1> sumB[K_TBINS];
    T sumT[K_TBINS];
    int cnt[K_TBINS];
    for (int k = 0; k < K_TBINS; ++k) { sumB[k].setZero(); sumT[k]=T(0); cnt[k]=0; }

    for (int i = 0; i < buf.n; ++i) {
      int k = 0;
      if (trange > T(1e-3)) {
        k = (int)floor((double)((buf.tempC[i] - tmin) / binW));
        k = clamp<int>(k, 0, K_TBINS - 1);
      }
      sumB[k] += buf.v[i];
      sumT[k] += buf.tempC[i];
      cnt[k]  += 1;
    }

    Eigen::Matrix<T,3,1> bcenters[K_TBINS];
    T tcenters[K_TBINS];
    int nb = 0;
    for (int k = 0; k < K_TBINS; ++k) {
      if (cnt[k] < 20) continue;
      bcenters[nb] = sumB[k] / (T)cnt[k];
      tcenters[nb] = sumT[k] / (T)cnt[k];
      nb++;
    }
    if (nb < 1) {
      last_fail_ = FitFail::TEMP_BINS_EMPTY;
      if (reason_out) *reason_out = last_fail_;
      return false;
    }

    TempBias3<T> biasT;
    if (nb >= 2) {
      biasT = fit_temp_bias3<T, K_TBINS>(bcenters, tcenters, nb, T0);
      if (!biasT.ok) {
        // fallback: still succeed with constant bias
        biasT.ok = true;
        biasT.T0 = T0;
        biasT.b0 = bcenters[0];
        biasT.k.setZero();
      }
    } else {
      biasT.ok = true;
      biasT.T0 = T0;
      biasT.b0 = bcenters[0];
      biasT.k.setZero();
    }

    out.ok = true;
    out.S = Eigen::Matrix<T,3,3>::Identity();
    out.biasT = biasT;

    last_fail_ = FitFail::OK;
    if (reason_out) *reason_out = FitFail::OK;
    return true;
  }
};

} // namespace imu_cal
