#ifndef KALMANF_H
#define KALMANF_H

#include <math.h>
#include <limits>

/*
   See: https://github.com/randyaliased/KalmANF/

   and

   See: R. Ali, T. van Waterschoot, "A frequency tracker based on a Kalman filter update of a single parameter adaptive notch filter", 
   Proceedings of the 26th International Conference on Digital Audio Effects (DAFx23), Copenhagen, Denmark, September 2023

   Conceptual model
   ----------------
   The algorithm tracks the dominant sinusoidal component of a real-valued signal y[n]
   by adapting the coefficient a in a second-order resonator / notch filter. The
   resonator is parameterized as

       s[n] = y[n] + ρ · a · s[n−1] − ρ² · s[n−2],

   where a = 2 cos(ω_d), and ω_d is the (digital) radian frequency per sample.

   The scalar state a is updated by a 1-D Kalman filter driven by the notch error e[n]:

       e[n] = s[n] − a[n−1] · s[n−1] + s[n−2],
       a[n] = a[n−1] + K[n] · e[n],

   with K[n] chosen according to a scalar Kalman update.

   Parameters (conceptual roles)
   -----------------------------
   • ρ (rho) – pole radius of the resonator, 0 < ρ < 1
       – ρ → 1.0  : high-Q, narrowband, long memory, more selective.
       – smaller ρ: more damping, broader bandwidth, less selective, more robust.

   • a – adaptive notch coefficient, a = 2 cos(ω_d)
       – Encodes the tracked digital frequency ω_d.
       – a ≈  2  ⇒ very low frequency (near DC).
       – a ≈  0  ⇒ mid-band (around f_s / 4).
       – a ≈ −2  ⇒ near Nyquist (f_s / 2).

   • q – process noise variance on a (Q ≈ q)
       – Controls how quickly the filter “forgets” its previous estimate of a.
       – Larger q: p grows faster between samples → larger Kalman gain → a adapts
         quickly (forgets history faster), but the frequency can become noisier.
       – Smaller q: p grows slowly → smaller Kalman gain → very smooth and
         “sticky” behaviour (remembers history longer), but slower to follow
         genuine frequency shifts.
         
   • r – measurement noise variance on e[n] (R ≈ r)
       – Models how noisy / unreliable the error e[n] is.
       – Larger r: smaller Kalman gain, smoother and slower updates.
       – Smaller r: larger Kalman gain, more aggressive and jitter-prone.

   • p – p_cov, Kalman error covariance on a
       – Internal state tracking the uncertainty in a.
       – Larger p → larger gain K (we believe a is uncertain).
       – Smaller p → smaller gain K (we believe a is well known).

   Mapping back to Hz
   ------------------
   Once the updated a is available, the instantaneous digital frequency is

       ω̂_d = arccos(a / 2),

   and the physical frequency in Hz is

       f̂ = (ω̂_d / Δt) / (2π),

   where Δt is the effective sample period. 
*/

template <typename Real = double>
class KalmANF {
private:
  static constexpr Real defaultRho = Real(0.995);
  static constexpr Real default_a  = Real(1.9999);

  class ANFResonator {
  public:
    Real s_prev1 = Real(0);
    Real s_prev2 = Real(0);
    Real a       = default_a;
    Real rho     = defaultRho;
    Real rho_sq  = defaultRho * defaultRho;

    void init(Real rho_init, Real a_init, Real s1, Real s2) {
      rho    = rho_init;
      rho_sq = rho * rho;
      a      = a_init;
      s_prev1 = s1;
      s_prev2 = s2;
    }

    Real compute_s(Real y) const {
      return y + rho * s_prev1 * a - rho_sq * s_prev2;
    }

    void update_state(Real s) {
      s_prev2 = s_prev1;
      s_prev1 = s;
    }

    Real get_phase() const {
      return std::atan2(s_prev1, s_prev2);
    }
  };

  ANFResonator res;

  // Kalman parameters
  Real p_cov = Real(1);
  Real q     = Real(1e-6);
  Real r     = Real(1e+3);

public:
  void init(Real rho      = defaultRho,
            Real q_       = Real(1e-6),
            Real r_       = Real(1e+3),
            Real p_cov_   = Real(1),
            Real s_prev1_ = Real(0),
            Real s_prev2_ = Real(0),
            Real a_       = default_a)
  {
    q     = q_;
    r     = r_;
    p_cov = p_cov_;
    res.init(rho, a_, s_prev1_, s_prev2_);
  }

  // Convenience initializer: seed a from a frequency guess in Hz.
  // f_guess_hz : initial frequency guess (Hz)
  // dt         : sample period (s)
  // rho, q_, r_, p_cov_ behave as in init().
  void initFromFreqGuess(Real f_guess_hz,
                         Real dt,
                         Real rho      = defaultRho,
                         Real q_       = Real(1e-6),
                         Real r_       = Real(1e+3),
                         Real p_cov_   = Real(1))
  {
    // ω_d (rad/sample) = 2π f / f_s = 2π f · dt
    const Real omega_d = Real(2) * Real(M_PI) * f_guess_hz * dt;
    const Real a_init  = Real(2) * std::cos(omega_d);
    init(rho, q_, r_, p_cov_, Real(0), Real(0), a_init);
  }

  // y: input sample, in *whatever* units (but tune q,r for that scale)
  // dt: actual sample period (seconds)
  Real process(Real y, Real dt, Real* e_out = nullptr) {
    Real delta_t = dt; 

    // 1. resonator
    Real s = res.compute_s(y);

    // 2. prediction
    p_cov += q;

    // 3. Kalman gain (original form)
    Real signal_power = res.s_prev1 * res.s_prev1;
    Real denom = signal_power + r / (p_cov + std::numeric_limits<Real>::epsilon());
    Real K = res.s_prev1 / (denom + Real(1e-12));

    // 4. error
    Real e = s - res.s_prev1 * res.a + res.s_prev2;

    // 5. update a
    Real a = res.a + K * e;

    // 6. keep a in acos domain
    if (a > Real(2) || a < Real(-2)) {
      a = (a > Real(2)) ? Real(1.99999) : Real(-1.99999);
    }

    // 7. covariance update
    p_cov = (Real(1) - K * res.s_prev1) * p_cov;
    p_cov = std::max(p_cov, Real(1e-12));

    // 8. frequency estimate
    Real omega_hat = std::acos(a / Real(2));  // rad/sample
    Real f_est = (omega_hat / delta_t) / (Real(2) * static_cast<Real>(M_PI)); // Hz

    // 9. state update
    res.a = a;
    res.update_state(s);

    if (e_out) *e_out = e;

    return f_est;
  }

  Real get_phase() const { return res.get_phase(); }
  Real get_a() const     { return res.a; }
  Real get_p_cov() const { return p_cov; }
};

#endif // KALMANF_H
