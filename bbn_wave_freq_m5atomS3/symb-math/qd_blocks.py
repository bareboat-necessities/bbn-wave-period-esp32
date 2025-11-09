# qd_blocks.py
# Block-by-block optimized LTI discretization (Phi, Qd)
#   - Attitude + gyro-bias (6x6)
#   - One OU axis [v,p,S,a] (4x4)  → replicate for x,y,z
#   - 3-axis linear block, uncorrelated OR correlated (via Σ ⊗ Qaxis_unit)
#   - Accel-bias random walk (3x3)
#
# Exact LTI Qd via Van Loan per block:
#   exp( [[F, LQcL^T],[0, -F^T]] h ) = [[Phi, S],[0, Phi^{-T}]],  Qd = S Phi^T

from sympy import symbols, Matrix, eye, diag, simplify, exp

# ---------------------------
# Shared symbols
# ---------------------------
h, tau = symbols('h tau', positive=True)
wx, wy, wz = symbols('w_x w_y w_z', real=True)

# Attitude/bias spectral densities
Qg_x, Qg_y, Qg_z = symbols('Qg_x Qg_y Qg_z', positive=True)
Qbg = symbols('Qbg', positive=True)

# Accel-bias RW spectral density
Qba = symbols('Qba', positive=True)

# OU stationary std per axis (uncorrelated case)
sigma_x, sigma_y, sigma_z = symbols('sigma_x sigma_y sigma_z', positive=True)

# ---------------------------
# Utilities
# ---------------------------
def skew(wx, wy, wz):
    return Matrix([[0,   -wz,  wy],
                   [wz,   0,  -wx],
                   [-wy, wx,   0]])

def vanloan_phi_qd(F: Matrix, L: Matrix, Qc: Matrix, h_sym):
    """Exact block discretization for LTI SDE using Van Loan."""
    n = F.shape[0]
    G = L * Qc * L.T
    A = Matrix.zeros(2*n, 2*n)
    A[:n, :n] = F
    A[:n, n:] = G
    A[n:, n:] = -F.T
    E = (A * h_sym).exp()
    Phi = simplify(E[:n, :n])
    S   = E[:n, n:]
    Qd  = simplify((S * Phi.T + (S * Phi.T).T) / 2)  # symmetrize
    return Phi, Qd

# ---------------------------
# Block 1: Attitude error (3) + gyro bias (3)
# dθ̇ = -[ω]× dθ - I b_g + (-I) n_g
# ḃ_g = 0               + ( I) w_bg
# ---------------------------
def phi_qd_attitude_bias(wx, wy, wz, h_sym, Qg_diag, Qbg_scalar):
    I3 = eye(3); Z3 = Matrix.zeros(3,3)
    W  = skew(wx, wy, wz)

    F_tt = -W
    F_tb = -I3
    F_bt = Z3
    F_bb = Z3

    F = Matrix([[*F_tt.row(0), *F_tb.row(0)],
                [*F_tt.row(1), *F_tb.row(1)],
                [*F_tt.row(2), *F_tb.row(2)],
                [*F_bt.row(0), *F_bb.row(0)],
                [*F_bt.row(1), *F_bb.row(1)],
                [*F_bt.row(2), *F_bb.row(2)]])

    # Noise map
    L = Matrix.zeros(6,6)
    L[:3,:3] = -I3    # gyro white noise into dθ
    L[3:,3:] =  I3    # gyro-bias RW into b_g

    Qc = diag(*Qg_diag, Qbg_scalar, Qbg_scalar, Qbg_scalar)
    return vanloan_phi_qd(F, L, Qc, h_sym)

# ---------------------------
# Block 2: One OU axis [v, p, S, a]
# v̇ = a; ṗ = v; Ṡ = p; ȧ = -(1/τ)a + √(2 σ²/τ) w
# ---------------------------
def phi_qd_ou_axis(tau_sym, sigma_sym, h_sym):
    F = Matrix([
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1/tau_sym]
    ])
    # Inject white noise only into 'a'
    L  = Matrix([[0],[0],[0],[1]])
    Qc = Matrix([[2 * sigma_sym**2 / tau_sym]])  # scalar spectral density
    return vanloan_phi_qd(F, L, Qc, h_sym)

# (Optional) Analytic Phi_axis (fast, matches your code) — drop-in if desired:
def phi_axis_analytic(h_sym, tau_sym):
    x = h_sym / tau_sym
    alpha = exp(-x)
    phi_va = -tau_sym * (alpha - 1)
    phi_pa = tau_sym**2 * (x + (alpha - 1))
    phi_Sa = tau_sym**3 * ((1/2)*x**2 - x - (alpha - 1))
    Phi = Matrix([[1,         0,        0,      phi_va],
                  [h_sym,     1,        0,      phi_pa],
                  [h_sym**2/2,h_sym,    1,      phi_Sa],
                  [0,         0,        0,      alpha ]])
    return simplify(Phi)

# ---------------------------
# Block 3a: Three-axis linear block (uncorrelated x/y/z)
# ---------------------------
def phi_qd_linear_3axis_uncorr(tau_sym, sig_x, sig_y, sig_z, h_sym):
    Phi_x, Qd_x = phi_qd_ou_axis(tau_sym, sig_x, h_sym)
    Phi_y, Qd_y = phi_qd_ou_axis(tau_sym, sig_y, h_sym)
    Phi_z, Qd_z = phi_qd_ou_axis(tau_sym, sig_z, h_sym)
    Phi_LL = diag(Phi_x, Phi_y, Phi_z)
    Qd_LL  = diag(Qd_x,  Qd_y,  Qd_z)
    return Phi_LL, Qd_LL

# ---------------------------
# Block 3b: Three-axis linear block (correlated via Σ)
# Idea: Phi_LL does not depend on Σ → blkdiag(Phi_axis,Phi_axis,Phi_axis)
#       Qd_LL = Σ ⊗ Qaxis_unit, where Qaxis_unit is Qd_axis with σ²=1.
# ---------------------------
def qd_linear_3axis_correlated(tau_sym, Sigma3, h_sym):
    # Compute unit-axis Qd (σ² = 1) once
    _, Qd_unit = phi_qd_ou_axis(tau_sym, 1, h_sym)  # 4x4
    # Kronecker assembly: 12x12
    Qd_LL = Matrix.zeros(12, 12)
    # Build Phi_LL (independent of Σ)
    Phi_axis = phi_axis_analytic(h_sym, tau_sym)     # faster than (F*h).exp()
    Phi_LL   = diag(Phi_axis, Phi_axis, Phi_axis)

    # Fill 3x3 block structure: each (i,j) block is Σ[i,j] * Qd_unit
    for i in range(3):
        for j in range(3):
            Qd_LL[4*i:4*(i+1), 4*j:4*(j+1)] = Sigma3[i,j] * Qd_unit
    # Symmetrize (numerical hygiene)
    Qd_LL = simplify((Qd_LL + Qd_LL.T) / 2)
    return Phi_LL, Qd_LL

# ---------------------------
# Block 4: Accel-bias RW (3x3)
# ḃ_a = w_ba,  with spectral density Qba (per axis)
# ---------------------------
def phi_qd_accel_bias(Qba_scalar, h_sym):
    F = Matrix.zeros(3,3)
    L = eye(3)
    Qc = Qba_scalar * eye(3)
    return vanloan_phi_qd(F, L, Qc, h_sym)

# ---------------------------
# Example assembly (kept lightweight; remove prints in production)
# ---------------------------
if __name__ == "__main__":
    # 1) Attitude + gyro-bias
    Phi_AA, Qd_AA = phi_qd_attitude_bias(
        wx, wy, wz, h,
        Qg_diag=(Qg_x, Qg_y, Qg_z),
        Qbg_scalar=Qbg
    )
    print("Phi_AA shape:", Phi_AA.shape, " Qd_AA shape:", Qd_AA.shape)

    # 2) Linear 3-axis (uncorrelated)
    Phi_LL_u, Qd_LL_u = phi_qd_linear_3axis_uncorr(tau, sigma_x, sigma_y, sigma_z, h)
    print("Phi_LL (uncorr) shape:", Phi_LL_u.shape, " Qd_LL (uncorr) shape:", Qd_LL_u.shape)

    # 3) Linear 3-axis (correlated Σ)
    sxx, syy, szz, sxz, syz = symbols('s_xx s_yy s_zz s_xz s_yz', real=True)
    Sigma3 = Matrix([[sxx,     0,   sxz],
                     [0,     syy,   syz],
                     [sxz,   syz,   szz]])
    Phi_LL_c, Qd_LL_c = qd_linear_3axis_correlated(tau, Sigma3, h)
    print("Phi_LL (corr) shape:", Phi_LL_c.shape, " Qd_LL (corr) shape:", Qd_LL_c.shape)

    # 4) Accel-bias RW
    Phi_ba, Qd_ba = phi_qd_accel_bias(Qba, h)
    print("Phi_ba shape:", Phi_ba.shape, " Qd_ba shape:", Qd_ba.shape)

    # 5) Full (optional): blkdiag the blocks you need
    Phi_full_uncorr = diag(Phi_AA, Phi_LL_u, Phi_ba)
    Qd_full_uncorr  = diag(Qd_AA,  Qd_LL_u,  Qd_ba)
    print("Full (uncorr) Phi shape:", Phi_full_uncorr.shape, " Qd shape:", Qd_full_uncorr.shape)

    Phi_full_corr = diag(Phi_AA, Phi_LL_c, Phi_ba)
    Qd_full_corr  = diag(Qd_AA,  Qd_LL_c,  Qd_ba)
    print("Full (corr)   Phi shape:", Phi_full_corr.shape, " Qd shape:", Qd_full_corr.shape)
