/*

Original python implementation of Fenton Wave:

import math
from numpy import (
    pi,
    cos,
    sin,
    zeros,
    arange,
    isfinite,
    newaxis,
    asarray,
    linspace,
    cosh,
    sinh,
    array,
    empty,
)
from numpy.linalg import solve
from numpy.fft import irfft

# If the air phase blending_height is None then the wave height times this
# default factor will be used
AIR_BLENDING_HEIGHT_FACTOR = 2


class RasciiError(Exception):
    pass


class NonConvergenceError(RasciiError):
    pass


def sinh_by_cosh(a, b):
    """
    A version of sinh(a)/cosh(b) where "b = a * f" and f is close
    to 1. This can then be written exp(a * (1 - f)) for large a
    """
    ans = np.zeros(a.size, float)
    for i, (ai, bi) in enumerate(zip(a, b)):
        if ai == 0:
            continue
        f = bi / ai
        if (ai > 30 and 0.5 < f < 1.5) or (ai > 200 and 0.1 < f < 1.9):
            ans[i] = math.exp(ai * (1 - f))
        else:
            sa = math.sinh(ai)
            cb = math.cosh(bi)
            ans[i] = sa / cb
    return ans


def cosh_by_cosh(a, b):
    """
    A version of cosh(a)/cosh(b) where "b = a * f" and f is close
    to 1. This can then be written exp(a * (1 - f)) for large a
    """
    ans = np.zeros(a.size, float)
    for i, (ai, bi) in enumerate(zip(a, b)):
        if ai == 0:
            ans[i] = 1.0 / math.cosh(bi)
            continue
        f = bi / ai
        if (ai > 30 and 0.5 < f < 1.5) or (ai > 200 and 0.1 < f < 1.9):
            ans[i] = math.exp(ai * (1 - f))
        else:
            ca = math.cosh(ai)
            cb = math.cosh(bi)
            ans[i] = ca / cb
    return ans




def trapezoid_integration(*argv, **kwargs):
    """
    Compatibility for numpy 2.0 rename of np.trapz to np.trapezoid
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(*argv, **kwargs)
    else:
        return np.trapz(*argv, **kwargs)


def np2py(val):
    """
    Convert a numpy array or numpy number into a list of Python floats
    or a single Python float

    We want the base types so that we can call repr() and get something
    that is pure Python (works in C++ after repr() conversion to code
    string) and does not include the string "np.float64" or similar
    (which repr() will in numpy 2.0)
    """
    if hasattr(val, "tolist"):
        return val.tolist()  # Convert numpy array to Python list of float
    elif hasattr(val, "item"):
        return val.item()  # Convert numpy float64 to Python float
    else:
        return val  # Assume this is allready a Python float


class FentonWave:
    required_input = {"height", "depth", "length", "N"}
    optional_input = {"air": None, "g": 9.81, "relax": 0.5}

    def __init__(
        self,
        height: float,
        depth: float,
        length: float | None = None,
        N: int = 5,
        period: float | None = None,
        air=None,
        g: float = 9.81,
        relax: float = 0.5,
    ):
        """
        Implement stream function waves based on the paper by Rienecker and
        Fenton (1981)

        * height: wave height above still water level
        * depth: still water distance from the flat sea bottom to the free surface
          in meters, but you can give -1.0 for infinite depth
        * length: the periodic length of the wave (optional, if not given then period is used)
        * N: the number of coefficients in the truncated Fourier series
        * period: the wave period (optional, if not given then length is used)
        """
        if length is None:
            if period is None:
                raise RasciiError("Either length or period must be given, both are None!")
            length = compute_length_from_period(
                height=height, depth=depth, period=period, N=N, g=g, relax=relax
            )

        self.height: float = height  #: The wave height
        self.depth: float = depth  #: The water depth
        self.length: float = length  #: The wave length
        self.order: int = N  #: The approximation order
        self.air = air  #: The optional air-phase model
        self.g: float = g  #: The acceleration of gravity
        self.relax: float = relax  #: The numerical relaxation in the optimization loop
        self.warnings: str = ""  #: Warnings raised when generating this wave

        # Find the coeffients through optimization
        data = fenton_coefficients(height, depth, length, N, g, relax=relax)
        self.set_data(data)

        # For evaluating velocities close to the free surface
        self.eta_eps: float = self.height / 1e5

        # Provide velocities also in the air phase
        if self.air is not None:
            self.air.set_wave(self)

    def set_data(self, data):
        """
        Update the coefficients defining this stream-function wave
        """
        self.data = data
        self.eta = data["eta"]  # Wave elevation at colocation points
        self.x = data["x"]  # Positions of colocation points
        self.k = data["k"]  # Wave number
        self.c = data["c"]  # Phase speed
        self.cs = self.c - data["Q"]  # Mean Stokes drift speed
        self.T = self.length / self.c  # Wave period
        self.omega = self.c * self.k  # Wave frequency

        # Cosine series coefficients for the elevation
        N = len(self.eta) - 1
        self.E = zeros(N + 1, float)
        J = arange(0, N + 1)
        self.E = trapezoid_integration(self.eta * cos(J * J[:, newaxis] * pi / N))


    def surface_elevation(self, x: float | list[float], t: float = 0.0, include_depth: bool = True):
        """
        Compute the surface elevation at time t for position(s) x
        """
        if isinstance(x, (float, int)):
            x = array([x], float)
        x = asarray(x)

        # Cosine transformation of the elevation
        N = len(self.eta) - 1
        J = arange(0, N + 1)
        k, c = self.k, self.c
        eta = 2 * trapezoid_integration(self.E * cos(J * k * (x[:, newaxis] - c * t))) / N

        if include_depth:
            if self.depth < 0:
                raise RasciiError("Cannot include depth in elevation for infinite depth")
            subtract = 0.0
        else:
            # Apply consistent water depth limitation with 'fenton_coefficients'
            subtract = 25 * self.length if self.depth < 0 else self.depth

        return eta - subtract


def fenton_coefficients(
    height, depth, length, N, g=9.8, maxiter=500, tolerance=1e-8, relax=1.0, num_steps=None
):
    """
    Find B, Q and R by Newton-Raphson following Rienecker and Fenton (1981)

    Using relaxation can help in some difficult cases, try a value less than 1
    to decrease convergence speed, but increase chances of converging.
    """
    if depth < 0:
        depth = 25 * length

    # Non dimensionalised input
    H = height / depth
    lam = length / depth
    k = 2 * pi / lam
    c = (math.tanh(k) / k) ** 0.5
    D = 1
    N_unknowns = 2 * (N + 1) + 2

    # Input data arrays
    J = arange(1, N + 1)
    M = arange(0, N + 1)
    x = M * lam / (2 * N)

    def initial_guess(H):
        """
        Initial guesses for the unknowns (linear wave)
        """
        B = zeros(N + 1, float)
        B[0] = c
        B[1] = -H / (4 * c * k)
        eta = 1 + H / 2 * cos(k * x)
        Q = c
        R = 1 + 0.5 * c**2
        return B, Q, R, eta

    def optimize(B, Q, R, eta, H):
        """
        Find B, Q and R by Newton iterations starting from the given initial
        guesses. According to Rienecker and Fenton (1981) a linear theory
        initial guess should work unless H close to breaking, then an initial
        guess from the optimization routine run with a slightly lower H should
        be used instead.
        """
        # Insert initial guesses into coefficient vector
        coeffs = zeros(N_unknowns, float)
        coeffs[: N + 1] = B
        coeffs[N + 1 : 2 * N + 2] = eta
        coeffs[2 * N + 2] = Q
        coeffs[2 * N + 3] = R
        f = func(coeffs, H, k, D, J, M)

        for it in range(1, maxiter + 1):
            jac = fprime(coeffs, H, k, D, J, M)
            delta = solve(jac, -f)
            coeffs += delta * relax
            f = func(coeffs, H, k, D, J, M)

            # Check the progress
            error = abs(f).max()
            eta_max = coeffs[N + 1 : 2 * N + 2].max()
            eta_min = coeffs[N + 1 : 2 * N + 2].min()
            if eta_max > 2:
                raise NonConvergenceError(
                    "Optimization did not converge. Got "
                    "max(eta)/depth = %r in iteration %d" % (eta_max, it)
                )
            elif eta_min < 0.1:
                raise NonConvergenceError(
                    "Optimization did not converge. Got "
                    "min(eta)/depth = %r in iteration %d" % (eta_min, it)
                )
            elif not isfinite(error):
                raise NonConvergenceError(
                    "Optimization did not converge. Got error %r in iteration %d" % (error, it)
                )
            elif error < tolerance:
                B = coeffs[: N + 1]
                eta = coeffs[N + 1 : 2 * N + 2]
                Q = coeffs[2 * N + 2]
                R = coeffs[2 * N + 3]
                return B, Q, R, eta, error, it
        raise NonConvergenceError(
            "Optimization did not converge after %d iterations, error = %r" % (it, error)
        )

    # Perform the optimization, optionally in steps gradually increasing H
    steps = wave_height_steps(num_steps, D, lam, H)
    B, Q, R, eta = initial_guess(steps[0])
    for Hi in steps:
        B, Q, R, eta, error, niter = optimize(B, Q, R, eta, Hi)

    # Scale back to physical space
    B[0] *= (g * depth) ** 0.5
    B[1:] *= (g * depth**3) ** 0.5
    return {
        "x": x * depth,
        "eta": eta * depth,
        "B": B,
        "Q": Q * (g * depth**3) ** 0.5,
        "R": R * g * depth,
        "k": k / depth,
        "c": B[0],
        "error": error,
        "niter": niter,
    }


def wave_height_steps(num_steps, D, lam, H):
    """
    Compute the breaking height and use this to select how many steps take when
    gradually increasing the wave height to improve convergence on high waves
    """
    # Breaking height
    Hb = 0.142 * math.tanh(2 * pi * D / lam) * lam

    # Try with progressively higher waves to get better initial conditions
    if num_steps is not None:
        pass
    if H > 0.75 * Hb:
        num_steps = 10
    elif H > 0.65 * Hb:
        num_steps = 5
    else:
        num_steps = 3

    if num_steps == 1:
        return [H]
    else:
        return linspace(H / num_steps, H, num_steps)


def func(coeffs, H, k, D, J, M):
    "The function to minimize"
    N_unknowns = coeffs.size
    N = J.size

    B0 = coeffs[0]
    B = coeffs[1 : N + 1]
    eta = coeffs[N + 1 : 2 * N + 2]
    Q = coeffs[2 * N + 2]
    R = coeffs[2 * N + 3]

    # The function to me minimized
    f = zeros(N_unknowns, float)

    # Loop over the N + 1 points along the half wave
    for m in M:
        S1 = sinh_by_cosh(J * k * eta[m], J * k * D)
        C1 = cosh_by_cosh(J * k * eta[m], J * k * D)
        S2 = sin(J * m * pi / N)
        C2 = cos(J * m * pi / N)

        # Velocity at the free surface
        # The sign of B0 is swapped from what is in the paper
        um = -B0 + k * J.dot(B * C1 * C2)
        vm = 0 + k * J.dot(B * S1 * S2)

        # Enforce a streamline along the free surface
        # The sign of B0 is swapped from what is in the paper
        f[m] = -B0 * eta[m] + B.dot(S1 * C2) + Q

        # Enforce the dynamic free surface boundary condition
        f[N + 1 + m] = (um**2 + vm**2) / 2 + eta[m] - R

    # Enforce mean(eta) = D
    f[-2] = trapezoid_integration(eta) / N - 1

    # Enforce eta_0 - eta_N = H, the wave height criterion
    f[-1] = eta[0] - eta[-1] - H

    return f



def fprime(coeffs, H, k, D, J, M):
    "The Jacobian of the function to minimize"
    N_unknowns = coeffs.size
    N = J.size

    jac = zeros((N_unknowns, N_unknowns), float)
    B0 = coeffs[0]
    B = coeffs[1 : N + 1]
    eta = coeffs[N + 1 : 2 * N + 2]

    for m in range(N + 1):
        S1 = sinh_by_cosh(J * k * eta[m], J * k * D)
        C1 = cosh_by_cosh(J * k * eta[m], J * k * D)
        S2 = sin(J * m * pi / N)
        C2 = cos(J * m * pi / N)

        SC = S1 * C2
        SS = S1 * S2
        CC = C1 * C2
        CS = C1 * S2

        # Velocity at the free surface
        um = -B0 + k * J.dot(B * CC)
        vm = 0 + k * J.dot(B * SS)

        # Derivatives of the eq. for the streamline along the free surface
        jac[m, N + 1 + m] = um
        jac[0 : N + 1, 0] = -eta
        jac[m, 1 : N + 1] = SC
        jac[m, -2] = 1

        # Derivatives of the dynamic free surface boundary condition
        jac[N + 1 + m, N + 1 + m] = 1 + (
            um * k**2 * B.dot(J**2 * SC) + vm * k**2 * B.dot(J**2 * CS)
        )
        jac[N + 1 + m, -1] = -1
        jac[N + 1 + m, 0] = -um
        jac[N + 1 + m, 1 : N + 1] = k * um * J * CC + k * vm * J * SS

    # Derivative of mean(eta) = 1
    jac[-2, N + 1 : 2 * N + 2] = M * 0 + 1 / N
    jac[-2, N + 1] = 1 / (2 * N)
    jac[-2, 2 * N + 1] = 1 / (2 * N)

    # Derivative of the wave height criterion
    jac[-1, N + 1] = 1
    jac[-1, 2 * N + 1] = -1

    return jac

def airy_compute_length_from_period(depth: float, period: float, g: float = 9.81) -> float:
    """
    Compute the wave length from the wave height, depth and period
    using the linear dispersion relation.
    """
    # Infinite depth approximation
    length = g * period**2 / (2 * pi)

    if depth < 0:
        # No need to compute the length via iterations for infinite depth
        return length

    # Find the length by Newton iterations
    length_1 = length * 0.95
    length_2 = length * 1.05
    T_1 = (length_1 * 2 * pi / (g * tanh(2 * pi * depth / length_1))) ** 0.5
    T_2 = (length_2 * 2 * pi / (g * tanh(2 * pi * depth / length_2))) ** 0.5
    length_N = 0.0
    iter = 0
    while abs(length_N - length) > 1e-4:
        # Store the previous length
        length = length_N

        # New guess for the wave length by interpolation
        f = (period - T_1) / (T_2 - T_1)
        length_N = length_1 + (length_2 - length_1) * f

        # Resulting period for the new length from the dispersion relation
        T_N = (length_N * 2 * pi / (g * tanh(2 * pi * depth / length_N))) ** 0.5

        # Update the two points used for the interpolation in the next iteration
        if T_N < period:
            length_1 = length_N
            T_1 = T_N
        else:
            length_2 = length_N
            T_2 = T_N

        iter += 1
        if iter > 100:
            raise NonConvergenceError(
                "Failed to converge when computing wave length from period for Airy waves"
            )

    return length_N


def compute_length_from_period(
    height: float,
    depth: float,
    period: float,
    N: int = 5,
    g: float = 9.81,
    relax: float = 0.5,
):
    """
    Compute the wave length from the wave period using the Fenton wave theory
    """

    # Initial guess is based on the linear dispersion relation for deep water waves
    length = airy_compute_length_from_period(depth=depth, period=period, g=g)

    # Find the length by Newton iterations
    wave1 = FentonWave(height=height, depth=depth, length=length * 0.95, N=N, g=g, relax=relax)
    wave2 = FentonWave(height=height, depth=depth, length=length * 1.05, N=N, g=g, relax=relax)

    length_N = 0.0
    iter = 0
    while abs(length_N - length) > 1e-4:
        # Store the previous length
        length = length_N

        # New guess for the wave length by interpolation
        f = (period - wave1.T) / (wave2.T - wave1.T)
        length_N = wave1.length + (wave2.length - wave1.length) * f

        # Resulting wave period for the new length from the dispersion relation
        waveN = FentonWave(height=height, depth=depth, length=length_N, N=N, g=g, relax=relax)

        # Update the two points used for the interpolation in the next iteration
        if waveN.T < period:
            wave1 = waveN
        else:
            wave2 = waveN

        iter += 1
        if iter > 100:
            raise NonConvergenceError(
                "Failed to converge when computing wave length from period for Fenton waves"
            )

    return length_N
    
*/

#pragma once

#include <ArduinoEigenDense.h>
#include <cmath>
#include <stdexcept>
#include <functional>
#include <limits>
#include <algorithm>

#ifdef FENTON_TEST
#include <iostream>
#include <fstream>
#endif

template <typename T>
constexpr const T& clamp_value(const T& val, const T& low, const T& high) {
    return (val < low) ? low : (val > high) ? high : val;
}

// Minimal Real FFT implementation compatible with Eigen
template <typename Scalar, int N>
class FentonFFT {
public:
    using Real = Scalar;
    using Complex = std::complex<Real>;
    using RealVec = Eigen::Matrix<Real, N, 1>;
    using ComplexVec = Eigen::Matrix<Complex, N, 1>;

    void forward(const RealVec& in, ComplexVec& out) {
        for (int k = 0; k < N; ++k) {
            Complex sum(0, 0);
            for (int n = 0; n < N; ++n) {
                Real angle = -2 * M_PI * k * n / N;
                sum += in(n) * Complex(std::cos(angle), std::sin(angle));
            }
            out(k) = sum;
        }
    }

    void inverse(const ComplexVec& in, RealVec& out) {
        for (int n = 0; n < N; ++n) {
            Complex sum(0, 0);
            for (int k = 0; k < N; ++k) {
                Real angle = 2 * M_PI * k * n / N;
                sum += in(k) * Complex(std::cos(angle), std::sin(angle));
            }
            out(n) = sum.real() / N;
        }
    }

    void inverse_cosine(const RealVec& coeffs, RealVec& signal) {
        const int M = N - 1;
        for (int i = 0; i < N; ++i) {
            Real sum = 0;
            for (int j = 0; j <= M; ++j) {
                Real phi = M_PI * j * i / M;
                Real w = (j == 0 || j == M) ? 0.5f : 1.0f;
                sum += coeffs(j) * std::cos(phi) * w;
            }
            signal(i) = 2.0f * sum / M;
        }
    }
};

// FentonWave class
template <int N = 4>
class FentonWave {
private:
    static constexpr int StateDim = 2 * (N + 1) + 2;

    using Real = float;
    using VectorF = Eigen::Matrix<Real, N + 1, 1>;
    using BigVector = Eigen::Matrix<Real, StateDim, 1>;
    using BigMatrix = Eigen::Matrix<Real, StateDim, StateDim>;

public:
    Real height, depth, length, g, relax;
    Real k, c, T, omega;
    VectorF eta, x, E, B;

    FentonWave(Real height, Real depth, Real length, Real g = 9.81f, Real relax = 0.5f)
        : height(height), depth(depth), length(length), g(g), relax(relax) {
        compute();
    }

    Real surface_elevation(Real x_val, Real t = 0) const {
        Real sum = 0;
        for (int j = 0; j <= N; ++j) {
            sum += E(j) * std::cos(j * k * (x_val - c * t));
        }
        return sum;
    }

    Real surface_slope(Real x, Real t = 0) const {
        Real d_eta = 0.0f;
        for (int j = 0; j <= N; ++j) {
            d_eta -= E(j) * j * k * std::sin(j * k * (x - c * t));
        }
        return d_eta;
    }

    Real surface_time_derivative(Real x, Real t = 0) const {
        return -c * surface_slope(x, t);
    }

    Real vertical_velocity(Real x, Real z, Real t = 0) const {
        Real w = 0.0f;
        for (int j = 1; j <= N; ++j) {
            Real kj = j * k;
            Real arg = kj * (x - c * t);
            Real denom = std::cosh(kj * depth);
            if (denom < std::numeric_limits<Real>::epsilon()) continue;
            Real term = B(j) * kj / denom;
            w += term * std::sin(arg) * std::sinh(kj * (z + depth));
        }
        return w;
    }

    // Getters
    Real get_c() const { return c; }
    Real get_k() const { return k; }
    Real get_T() const { return T; }
    Real get_omega() const { return omega; }
    Real get_length() const { return length; }
    Real get_height() const { return height; }
    const VectorF& get_eta() const { return eta; }

private:
    void compute() {
        if (depth < 0) depth = 25.0f * length;
        Real H = height / depth;
        Real lam = length / depth;
        k = 2 * M_PI / lam;
        Real D = 1.0f;
        Real kc = k;
        Real c0 = std::sqrt(std::tanh(kc) / kc);

        VectorF x_nd;
        for (int m = 0; m <= N; ++m)
            x_nd(m) = lam * m / (2.0f * N);

        B.setZero();
        B(0) = c0;
        B(1) = -H / (4.0f * c0 * k);

        VectorF eta_nd;
        for (int m = 0; m <= N; ++m)
            eta_nd(m) = 1.0f + H / 2.0f * std::cos(k * x_nd(m));

        Real Q = c0, R = 1.0f + 0.5f * c0 * c0;

        for (Real Hi : wave_height_steps(H, D, lam)) {
            optimize(B, Q, R, eta_nd, Hi, k, D);
        }

        Real sqrt_gd = std::sqrt(g * depth);
        B(0) *= sqrt_gd;
        for (int j = 1; j <= N; ++j)
            B(j) *= std::sqrt(g * std::pow(depth, 3));
        Q *= std::sqrt(g * std::pow(depth, 3));
        R *= g * depth;

        for (int i = 0; i <= N; ++i) {
            x(i) = x_nd(i) * depth;
            eta(i) = eta_nd(i) * depth;
        }

        k = k / depth;
        c = B(0);
        T = length / c;
        omega = c * k;

        // Use irfft-like logic to compute E coefficients
        FentonFFT<Real, N + 1> fft;
        fft.inverse_cosine(eta, E);
    }

    std::array<Real, 10> wave_height_steps(Real H, Real D, Real lam) {
        Real Hb = 0.142f * std::tanh(2 * M_PI * D / lam) * lam;
        int num = (H > 0.75f * Hb) ? 10 : (H > 0.65f * Hb) ? 5 : 3;
        std::array<Real, 10> steps{};
        for (int i = 0; i < num; ++i)
            steps[i] = H * (i + 1) / num;
        return steps;
    }

    void optimize(VectorF& B, Real& Q, Real& R,
                  VectorF& eta, Real H, Real k, Real D)
    {
        constexpr int NU = 2 * (N + 1) + 2;
        Eigen::Matrix<Real, NU, 1> coeffs;
        coeffs.template segment<N + 1>(0) = B;
        coeffs.template segment<N + 1>(N + 1) = eta;
        coeffs(2 * N + 2) = Q;
        coeffs(2 * N + 3) = R;

        for (int iter = 0; iter < 100; ++iter) {
            Eigen::Matrix<Real, NU, 1> f = compute_residual(coeffs, H, k, D);
            Eigen::Matrix<Real, NU, NU> J = compute_jacobian(coeffs, H, k, D);
            Eigen::Matrix<Real, NU, 1> delta = J.fullPivLu().solve(-f);
            coeffs += relax * delta;

            if (f.cwiseAbs().maxCoeff() < 1e-8f) break;
        }

        B = coeffs.template segment<N + 1>(0);
        eta = coeffs.template segment<N + 1>(N + 1);
        Q = coeffs(2 * N + 2);
        R = coeffs(2 * N + 3);
    }

    Eigen::Matrix<Real, 2 * (N + 1) + 2, 1>
    compute_residual(const Eigen::Matrix<Real, 2 * (N + 1) + 2, 1>& coeffs, Real H, Real k, Real D) {
        constexpr int NU = 2 * (N + 1) + 2;
        Eigen::Matrix<Real, NU, 1> f;
        auto B = coeffs.template segment<N + 1>(0);
        auto eta = coeffs.template segment<N + 1>(N + 1);
        Real Q = coeffs(2 * N + 2);
        Real R = coeffs(2 * N + 3);
        Real B0 = B(0);

        for (int m = 0; m <= N; ++m) {
            Real x_m = M_PI * m / N;
            Real eta_m = eta(m);
            Real um = -B0, vm = 0, S1, C1, S2, C2;
            for (int j = 1; j <= N; ++j) {
                Real aj = j * k * eta_m;
                Real bj = j * k * D;
                Real sinh_ratio = std::sinh(aj) / std::cosh(bj);
                Real cosh_ratio = std::cosh(aj) / std::cosh(bj);
                S1 = sinh_ratio;
                C1 = cosh_ratio;
                S2 = std::sin(j * x_m);
                C2 = std::cos(j * x_m);
                um += k * j * B(j) * C1 * C2;
                vm += k * j * B(j) * S1 * S2;
            }
            f(m) = -B0 * eta_m;
            for (int j = 1; j <= N; ++j) {
                Real aj = j * k * eta_m;
                Real bj = j * k * D;
                Real sinh_ratio = std::sinh(aj) / std::cosh(bj);
                Real C2 = std::cos(j * M_PI * m / N);
                f(m) += B(j) * sinh_ratio * C2;
            }
            f(m) += Q;
            f(N + 1 + m) = 0.5f * (um * um + vm * vm) + eta_m - R;
        }

        // mean(eta) = 1
        f(2 * N + 2) = eta.sum() / N - 1.0f;

        // eta(0) - eta(N) = H
        f(2 * N + 3) = eta(0) - eta(N) - H;

        return f;
    }

    Eigen::Matrix<Real, 2 * (N + 1) + 2, 2 * (N + 1) + 2>
    compute_jacobian(const Eigen::Matrix<Real, 2 * (N + 1) + 2, 1>& coeffs, Real H, Real k, Real D) {
        constexpr int NU = 2 * (N + 1) + 2;
        Eigen::Matrix<Real, NU, NU> J = Eigen::Matrix<Real, NU, NU>::Zero();
        auto B = coeffs.template segment<N + 1>(0);
        auto eta = coeffs.template segment<N + 1>(N + 1);
        Real B0 = B(0);

        for (int m = 0; m <= N; ++m) {
            Real eta_m = eta(m);
            Real x_m = M_PI * m / N;
            Real um = -B0, vm = 0;

            for (int j = 1; j <= N; ++j) {
                Real aj = j * k * eta_m;
                Real bj = j * k * D;
                Real sinh_ratio = std::sinh(aj) / std::cosh(bj);
                Real cosh_ratio = std::cosh(aj) / std::cosh(bj);
                Real s2 = std::sin(j * x_m);
                Real c2 = std::cos(j * x_m);
                um += k * j * B(j) * cosh_ratio * c2;
                vm += k * j * B(j) * sinh_ratio * s2;
            }

            J(m, 0) = -eta_m;
            for (int j = 1; j <= N; ++j) {
                Real aj = j * k * eta_m;
                Real bj = j * k * D;
                Real sinh_ratio = std::sinh(aj) / std::cosh(bj);
                Real c2 = std::cos(j * M_PI * m / N);
                J(m, j) = sinh_ratio * c2;
            }
            J(m, N + 1 + m) = -B0;
            J(m, 2 * N + 2) = 1;

            J(N + 1 + m, N + 1 + m) = 1.0f;
            J(N + 1 + m, 0) = -um;
        }

        for (int j = 0; j <= N; ++j)
            J(2 * N + 2, N + 1 + j) = 1.0f / N;

        J(2 * N + 3, N + 1) = 1;
        J(2 * N + 3, 2 * N + 1) = -1;

        return J;
    }
};


/**
 * @brief Class for tracking vertical kinematics of a floating object on a nonlinear wave surface.
 */
template<int N = 4>
class WaveSurfaceTracker {
private:
    FentonWave<N> wave;

    float t = 0.0f;
    float x = 0.0f;
    float dt = 0.005f;

    float prev_z = 0.0f;
    float prev_dzdt = 0.0f;

    static constexpr float slope_eps = 1e-6f;  // Prevent division by zero

    float mean_eta = 0.0f;

    // Robust periodic wrapping for horizontal position x
    float wrap_periodic(float val, float period) const {
        while (val < 0.0f) val += period;
        while (val >= period) val -= period;
        return val;
    }

    // Compute mean elevation offset by sampling wave surface at t=0
    void compute_mean_elevation(int samples = 100) {
        float sum = 0.0f;
        float L = wave.get_length();
        for (int i = 0; i < samples; ++i) {
            float xi = L * i / static_cast<float>(samples - 1);
            sum += wave.surface_elevation(xi, 0.0f);
        }
        mean_eta = sum / static_cast<float>(samples);
    }

    // Compute horizontal speed dx/dt using kinematic constraint
    float compute_horizontal_speed(float x_pos, float time) const {
        float eta      = wave.surface_elevation(x_pos, time) - mean_eta;
        float eta_dot  = wave.surface_time_derivative(x_pos, time);
        float eta_x    = wave.surface_slope(x_pos, time);
        // Pass physical elevation (add mean_eta back) for vertical velocity
        float w        = wave.vertical_velocity(x_pos, eta + mean_eta, time);

        // Clamp slope to avoid division by zero or extreme values
        if (std::abs(eta_x) < slope_eps)
            eta_x = (eta_x >= 0.0f) ? slope_eps : -slope_eps;

        return (w - eta_dot) / eta_x;
    }

    // 4th-order Runge-Kutta integration for horizontal position
    float rk4_integrate_x(float x_curr, float t_curr, float dt_step) const {
        float k1 = compute_horizontal_speed(x_curr, t_curr);
        float k2 = compute_horizontal_speed(x_curr + 0.5f * dt_step * k1, t_curr + 0.5f * dt_step);
        float k3 = compute_horizontal_speed(x_curr + 0.5f * dt_step * k2, t_curr + 0.5f * dt_step);
        float k4 = compute_horizontal_speed(x_curr + dt_step * k3, t_curr + dt_step);
        return x_curr + (dt_step / 6.0f) * (k1 + 2.0f*k2 + 2.0f*k3 + k4);
    }

public:
    WaveSurfaceTracker(float height, float depth, float length)
        : wave(height, depth, length) {
        compute_mean_elevation();
    }

    /**
     * @brief Track vertical motion of a floating object on the wave surface.
     * 
     * @param duration  Total simulation time (seconds)
     * @param timestep  Integration time step (seconds)
     * @param callback  Function to call at each time step:
     *                  void callback(t, z, dzdt, ddzdt2, x);
     */
    void track_floating_object(
        float duration,
        float timestep,
        std::function<void(float, float, float, float, float)> callback)
    {
        const float wave_T = wave.get_T();
        const float wave_L = wave.get_length();

        dt = std::clamp(timestep, 1e-5f, 0.2f * wave_T / 20.0f);

        t = 0.0f;
        x = 0.0f;

        // Initialize previous vertical position and vertical velocity properly
        prev_z = wave.surface_elevation(x, t) - mean_eta;
        prev_dzdt = wave.vertical_velocity(x, prev_z + mean_eta, t);

        while (t <= duration) {
            // RK4 step for horizontal position
            float x_next = rk4_integrate_x(x, t, dt);

            // Periodicity wrap
            x_next = wrap_periodic(x_next, wave_L);

            t += dt;
            x = x_next;

            // Surface elevation at new x and t (zero-centered)
            float z = wave.surface_elevation(x, t) - mean_eta;

            // Vertical velocity and acceleration by finite difference
            float dzdt = (z - prev_z) / dt;
            float ddzdt2 = (dzdt - prev_dzdt) / dt;

            callback(t, z, dzdt, ddzdt2, x);

            prev_z = z;
            prev_dzdt = dzdt;
        }
    }
};


#ifdef FENTON_TEST
template class FentonWave<4>;
template class WaveSurfaceTracker<4>;

void FentonWave_test_1() {
    const float height = 2.0f;
    const float depth = 10.0f;
    const float length = 50.0f;

    FentonWave<4> wave(height, depth, length);

    std::ofstream out("wave_data.csv");
    out << "x,elevation\n";
    for (float x = 0; x <= length; x += 0.05f) {
        float eta = wave.surface_elevation(x, 0);
        out << x << "," << eta << "\n";
    }
    std::cerr << "Expected wave length: " << length << "\n";
    std::cerr << "Computed wave length: " << 2 * M_PI / wave.get_k() << "\n";
}

void FentonWave_test_2() {
    // Wave parameters
    const float height = 2.0f;   // Wave height (m)
    const float depth = 10.0f;   // Water depth (m)
    const float length = 50.0f;  // Wavelength (m)
    
    // Simulation parameters
    const float duration = 20.0f; // Simulation duration (s)
    const float dt = 0.05f;         // Time step (s)

    // Create a 4th-order Fenton wave and a surface tracker
    WaveSurfaceTracker<4> tracker(height, depth, length);

    // Output file
    std::ofstream out("wave_tracker_data.csv");
    out << "Time(s),Displacement(m),Velocity(m/s),Acceleration(m/s²),X_Position(m)\n";

    // Define the kinematics callback (writes data to file)
    auto kinematics_callback = [&out](
        float time, float elevation, float vertical_velocity, float vertical_acceleration, float horizontal_position) {
        out << time << "," << elevation << "," << vertical_velocity << "," << vertical_acceleration << "," << horizontal_position << "\n";
    };

    // Track floating object (using callback)
    tracker.track_floating_object(duration, dt, kinematics_callback);
}

#endif


