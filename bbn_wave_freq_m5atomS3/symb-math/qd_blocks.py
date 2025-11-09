# qd_blocks_symbolic.py
# Block-by-block symbolic discretization using closed forms (fast)
#
# Blocks:
#   1) Attitude error (3) + gyro bias (3): exact Phi from Rodrigues; Qd via symbolic 3x3 integrals
#   2) OU axis [v,p,S,a] (4x4): analytic Phi and Qd using your closed-form K(h)
#   3) Linear 3-axis: uncorrelated (blkdiag) or correlated (Σ ⊗ Q_axis_unit)
#   4) Accel-bias RW (3x3): Phi = I, Qd = q_ba * h * I
#
# Assumptions for closed-form attitude Qd:
#   Q_g  = q_g  * I_3  (gyro white noise isotropic)
#   Q_bg = q_bg * I_3  (gyro-bias RW isotropic)
#
# Python 3.x, SymPy ≥ 1.10
# ---------------------------------------------------------------

from sympy import (
    symbols, Matrix, eye, diag, simplify, sqrt, sin, cos, exp, Rational
)

# ---------------------------
# Shared symbols
# ---------------------------
h, tau = symbols('h tau', positive=True)
wx, wy, wz = symbols('w_x w_y w_z', real=True)

# OU stationary std per axis
sigma_x, sigma_y, sigma_z = symbols('sigma_x sigma_y sigma_z', positive=True)

# Attitude/bias spectral densities (assume isotropic for closed forms)
qg, qbg = symbols('q_g q_bg', positive=True)     # gyro white-noise, gyro-bias RW
qba      = symbols('q_ba', positive=True)        # accel-bias RW

# ---------------------------
# Helpers
# ---------------------------
def skew(wx, wy, wz):
    return Matrix([[0,   -wz,  wy],
                   [wz,   0,  -wx],
                   [-wy, wx,   0]])

def rodrigues_R(wx, wy, wz, t):
    """
    R(t) = exp(-[w]_x t) = I - sin(theta) K + (1 - cos(theta)) K^2,
    where K = [u]_x, u = w/||w||, theta = ||w|| t.
    """
    W  = skew(wx, wy, wz)
    wn = sqrt(wx**2 + wy**2 + wz**2)
    I3 = eye(3)
    # Use W/wn symbolically; for wn→0 the series limit applies
    K = W / wn
    theta = wn * t
    return simplify(I3 - sin(theta)*K + (1 - cos(theta))*(K*K))

def B_of_t(wx, wy, wz, t):
    """
    B(t) = - ∫_0^t R(s) ds
         = -[ t I - (1 - cos θ)/||w||^2 W + (t - sin θ / ||w||)/||w||^2 W^2 ],
    with θ = ||w|| t, W = [w]_x.
    """
    W  = skew(wx, wy, wz)
    wn = sqrt(wx**2 + wy**2 + wz**2)
    I3 = eye(3)
    theta = wn * t
    invw2 = 1 / (wn**2)
    term1 = I3 * t
    term2 = (1 - cos(theta)) * invw2 * W
    term3 = (t - sin(theta)/wn) * invw2 * (W*W)
    return simplify(-(term1 - term2 + term3))

def IB_0_h(wx, wy, wz, t):
    """
    ∫_0^t B(s) ds (closed form).
    IB = -[ 1/2 t^2 I - ((t - sinθ/||w||)/||w||^2) W
            + ((1/2 t^2) + (cosθ - 1)/||w||^2)/||w||^2 W^2 ]
    """
    W  = skew(wx, wy, wz)
    wn = sqrt(wx**2 + wy**2 + wz**2)
    I3 = eye(3)
    theta = wn * t
    invw2 = 1/(wn**2)
    invw4 = invw2**2
    termI  = I3 * (t**2/2)
    termW  = ((t - sin(theta)/wn) * invw2) * W
    termW2 = ((t**2/2) + (cos(theta) - 1)*invw2) * invw2 * (W*W)
    return simplify(-(termI - termW + termW2))

# ---------------------------
# OU axis: analytic Phi and Qd
# State order: [v, p, S, a]
# ---------------------------
def Phi_axis_analytic(h, tau):
    x = h / tau
    alpha = exp(-x)  # e^{-x}
    phi_va = -tau * (alpha - 1)
    phi_pa = tau**2 * (x + (alpha - 1))
    phi_Sa = tau**3 * (x**2 / 2 - x - (alpha - 1))
    return simplify(Matrix([
        [1,        0,        0,      phi_va],
        [h,        1,        0,      phi_pa],
        [h**2/2,   h,        1,      phi_Sa],
        [0,        0,        0,      alpha ]
    ]))

def Qd_axis_closed(h, tau, sigma):
    """
    Qd_axis = q_c * Sym(K(h)),  q_c = 2*sigma^2/tau.
    Closed-form K entries (matches your implementation).
    """
    x = h / tau
    alpha  = exp(-x)           # e^{-x}
    em1    = alpha - 1         # e^{-x} - 1 (negative)
    em1_2  = exp(-2*x) - 1     # e^{-2x} - 1 (negative)

    tau2 = tau**2
    tau3 = tau**3
    A0 = -tau * em1
    A1 = tau2 * (-em1 - x * alpha)
    A2 = tau3 * (-2*em1 + alpha * x * (x + 2))
    B0 = -(tau/2) * em1_2

    C0 = h
    C1 = h**2 / 2
    C2 = h**3 / 3
    C3 = h**4 / 4
    C4 = h**5 / 5

    I1mA0   = C0 - A0
    Ix1mA1  = C1 - A1
    Ix21mA2 = C2 - A2

    K_vv = tau2*(C0 - 2*A0 + B0)
    K_pv = tau2*Ix1mA1 - tau3*I1mA0 + tau3*(A0 - B0)
    K_Sv = (tau**2/2)*Ix21mA2 - tau**3*Ix1mA1 + tau**4*I1mA0 - tau**4*(A0 - B0)
    K_aa = B0
    K_va = tau*(A0 - B0)
    K_pa = tau*A1 - tau2*A0 + tau2*B0
    K_Sa = (tau/2)*A2 - tau**2*A1 + tau**3*A0 - tau**3*B0
    K_pp = tau2*C2 - 2*tau**3*C1 + 2*tau**3*A1 + tau**4*C0 - 2*tau**4*A0 + tau**4*B0
    K_Sp = (tau**2/2)*C3 - (3*tau**3/2)*C2 + 2*tau**4*C1 - tau**5*C0 + \
           (tau**3/2)*A2 - 2*tau**4*A1 + 2*tau**5*A0 - tau**5*B0
    K_SS = (tau**2/4)*C4 - tau**3*C3 + 2*tau**4*C2 - 2*tau**5*C1 + tau**6*C0 \
           - tau**4*A2 + 2*tau**5*A1 - 2*tau**6*A0 + tau**6*B0

    K = Matrix([
        [K_vv, K_pv, K_Sv, K_va],
        [K_pv, K_pp, K_Sp, K_pa],
        [K_Sv, K_Sp, K_SS, K_Sa],
        [K_va, K_pa, K_Sa, K_aa]
    ])
    qc = 2 * sigma**2 / tau
    Qd = simplify(qc * (K + K.T) / 2)
    return Qd

# ---------------------------
# Linear 3-axis blocks
# ---------------------------
def Phi_Qd_linear_uncorrelated(h, tau, sigx, sigy, sigz):
    Phi_axis = Phi_axis_analytic(h, tau)
    Qd_x = Qd_axis_closed(h, tau, sigx)
    Qd_y = Qd_axis_closed(h, tau, sigy)
    Qd_z = Qd_axis_closed(h, tau, sigz)
    Phi_LL = diag(Phi_axis, Phi_axis, Phi_axis)
    Qd_LL  = diag(Qd_x, Qd_y, Qd_z)
    return simplify(Phi_LL), simplify(Qd_LL)

def Phi_Qd_linear_correlated(h, tau, Sigma3):
    """
    Correlated stationary covariance across axes:
      Q_LL = Σ ⊗ Q_axis_unit,   with Q_axis_unit = Qd_axis_closed(h,tau, sigma=1).
    Phi_LL is independent of Σ.
    """
    Phi_axis = Phi_axis_analytic(h, tau)
    Qd_unit  = Qd_axis_closed(h, tau, 1)
    Qd_LL = Matrix.zeros(12, 12)
    for i in range(3):
        for j in range(3):
            Qd_LL[4*i:4*(i+1), 4*j:4*(j+1)] = simplify(Sigma3[i,j] * Qd_unit)
    Phi_LL = diag(Phi_axis, Phi_axis, Phi_axis)
    return simplify(Phi_LL), simplify((Qd_LL + Qd_LL.T) / 2)

# ---------------------------
# Attitude + gyro-bias (6x6): exact Phi, symbolic Qd
# ---------------------------
def Phi_Qd_attitude_bias(h, wx, wy, wz, qg_iso, qbg_iso):
    """
    Assumes Q_g = qg_iso * I_3 and Q_bg = qbg_iso * I_3.
    Returns Phi_AA (exact) and Qd_AA symbolically.
    """
    I3 = eye(3)

    # Phi upper-left: R(h); upper-right: B(h)
    R = rodrigues_R(wx, wy, wz, h)
    B = B_of_t(wx, wy, wz, h)
    Phi_AA = Matrix([[*R.row(0), *B.row(0)],
                     [*R.row(1), *B.row(1)],
                     [*R.row(2), *B.row(2)],
                     [0,0,0, 1,0,0],
                     [0,0,0, 0,1,0],
                     [0,0,0, 0,0,1]])
    Phi_AA = simplify(Phi_AA)

    # Q blocks:
    # Q_tt = ∫ R(s) qg I R(s)^T ds + ∫ B(s) qbg I B(s)^T ds
    #      = qg * h * I + qbg * ∫ B(s)B(s)^T ds
    Q_tt_R = qg_iso * h * I3

    # Compute ∫_0^h B(s) B(s)^T ds elementwise (3x3)
    s = symbols('s', real=True, nonnegative=True)
    Bs = B_of_t(wx, wy, wz, s)
    integrand = Bs * Bs.T
    Q_tt_B = Matrix([[ simplify(qbg_iso * integrand[i,j].integrate((s, 0, h)))
                       for j in range(3)] for i in range(3)])
    Q_tt = simplify(Q_tt_R + Q_tt_B)

    # Q_tb = (∫_0^h B(s) ds) qbg I
    IB = IB_0_h(wx, wy, wz, h)
    Q_tb = simplify(IB * (qbg_iso * I3))

    # Q_bb = qbg * h I
    Q_bb = qbg_iso * h * I3

    # Assemble symmetric 6x6
    top = Q_tt.row_join(Q_tb)
    bot = Q_tb.T.row_join(Q_bb)
    Qd_AA = simplify(Matrix.vstack(top, bot))
    return Phi_AA, Qd_AA

# ---------------------------
# Accel-bias RW (3x3)
# ---------------------------
def Phi_Qd_accel_bias(h, qba_iso):
    Phi = eye(3)
    Qd  = qba_iso * h * eye(3)
    return Phi, Qd

# ---------------------------
# Demo / Assembly
# ---------------------------
if __name__ == "__main__":
    # Linear block (uncorrelated)
    Phi_LL_u, Qd_LL_u = Phi_Qd_linear_uncorrelated(h, tau, sigma_x, sigma_y, sigma_z)
    print("Phi_LL (uncorr) shape:", Phi_LL_u.shape, "Qd_LL (uncorr) shape:", Qd_LL_u.shape)

    # Linear block (correlated) with symbolic Sigma3
    sxx, syy, szz, sxz, syz = symbols('s_xx s_yy s_zz s_xz s_yz', real=True)
    Sigma3 = Matrix([[sxx, 0,   sxz],
                     [0,   syy, syz],
                     [sxz, syz, szz]])
    Phi_LL_c, Qd_LL_c = Phi_Qd_linear_correlated(h, tau, Sigma3)
    print("Phi_LL (corr)   shape:", Phi_LL_c.shape, "Qd_LL (corr)   shape:", Qd_LL_c.shape)

    # Attitude + gyro bias
    Phi_AA, Qd_AA = Phi_Qd_attitude_bias(h, wx, wy, wz, qg, qbg)
    print("Phi_AA shape:", Phi_AA.shape, "Qd_AA shape:", Qd_AA.shape)

    # Accel-bias
    Phi_ba, Qd_ba = Phi_Qd_accel_bias(h, qba)
    print("Phi_ba  shape:", Phi_ba.shape, "Qd_ba  shape:", Qd_ba.shape)

    # Full (uncorrelated OU)
    Phi_full = diag(Phi_AA, Phi_LL_u, Phi_ba)
    Qd_full  = diag(Qd_AA,  Qd_LL_u,  Qd_ba)
    print("Phi_full shape:", Phi_full.shape, "Qd_full shape:", Qd_full.shape)
