# block-phi-qd-sym.py
# SymPy derivations of Phi(h) and Qd(h) from F,L block structure.
# Fast & fully symbolic for:
#   (1) OU axis [v,p,S | a]
#   (2) Attitude + gyro-bias [dθ | b_g] with isotropic Q_g, Q_bg

import sympy as sp

# ---------- pretty helpers ----------
def pp(title, M):
    print("\n" + title + "\n" + "-"*len(title))
    sp.pprint(sp.simplify(M), wrap_line=False, num_columns=140)

def to_latex_block(title, M):
    print("\n%% " + title)
    print(sp.latex(sp.simplify(M)))

# ---------- OU axis: [v,p,S | a] ----------
def ou_axis_phi_qd(h, tau, sigma2):
    """
    State order: x = [v, p, S, a]^T
      dv = a dt,   dp = v dt,   dS = p dt,
      da = -(1/tau)a dt + w_a,   with stationary var sigma2 -> q_c = 2*sigma2/tau
    F = [[A,B],[0,D]], A(3x3) nilpotent chain, B = [1,0,0]^T, D = [-1/tau]
    L has only the 'a' channel here -> r=1, Qc = [ 2*sigma2/tau ].
    """
    t, s = sp.symbols('t s', positive=True, real=True)

    # A, B, D
    A = sp.Matrix([[0,0,0],
                   [1,0,0],
                   [0,1,0]])
    B = sp.Matrix([[1],[0],[0]])
    D = sp.Matrix([[-1/tau]])

    # e^{A s} from nilpotent series: A^3=0
    I3 = sp.eye(3); A2 = A*A
    eAs = I3 + A*s + A2*(s**2/2)

    # e^{D s} is scalar exp
    eDs = sp.exp(-s/tau)

    # Phi blocks
    Phi_TL = (I3 + A*h + A2*(h**2/2))             # e^{A h}
    # Phi_TR = ∫_0^h e^{A s} B e^{D s} ds  (3x1)
    integrand_TR = eAs * B * eDs
    Phi_TR = sp.Matrix(3,1, lambda i,j: sp.integrate(integrand_TR[i,j], (s,0,h)))
    Phi_BR = sp.exp(-h/tau)                        # scalar

    # Assemble Phi 4x4
    Phi = sp.eye(4)
    Phi[0:3,0:3] = sp.simplify(Phi_TL)
    Phi[0:3,3:4] = sp.simplify(Phi_TR)
    Phi[3,3]     = sp.simplify(Phi_BR)

    # Qd = ∫_0^h C(s) Qc C(s)^T ds
    # Here L_top has 0 cols, L_bot = [1]; so C_top = J(s), C_bot = e^{Ds}
    # with J(s) = ∫_0^s e^{A u} B e^{D u} du  (3x1).
    u = sp.symbols('u', positive=True, real=True)
    eAu = I3 + A*u + A2*(u**2/2)
    eDu = sp.exp(-u/tau)
    J_s = sp.Matrix(3,1, lambda i,j: sp.integrate((eAu*B*eDu)[i,0], (u,0,s)))
    C_top = J_s
    C_bot = sp.Matrix([[eDs]])   # 1x1

    qc = 2*sigma2/tau
    Qc = sp.Matrix([[qc]])       # (1x1)

    # integrand = [C_top; C_bot] * qc * [C_top^T, C_bot^T]
    C_top_simpl = sp.simplify(C_top)
    integrand_Q = sp.zeros(4,4)
    # Top-left 3x3
    integrand_Q[0:3,0:3] = qc * (C_top_simpl * C_top_simpl.T)
    # Cross 3x1 and 1x3
    integrand_Q[0:3,3:4] = qc * (C_top_simpl * C_bot.T)
    integrand_Q[3:4,0:3] = qc * (C_bot * C_top_simpl.T)
    # Bottom-right 1x1
    integrand_Q[3,3]     = qc * (C_bot * C_bot.T)[0,0]

    Qd = sp.Matrix(4,4, lambda i,j: sp.integrate(integrand_Q[i,j], (s,0,h)))
    Qd = sp.simplify((Qd + Qd.T)/2)

    return sp.simplify(Phi), sp.simplify(Qd)

# ---------- Attitude + gyro-bias: [dθ | b_g] ----------
def skew(wx, wy, wz):
    return sp.Matrix([[0, -wz,  wy],
                      [wz,  0, -wx],
                      [-wy, wx,  0]])

def rodrigues_R(wx, wy, wz, s):
    W  = skew(wx, wy, wz)
    wn = sp.sqrt(wx**2 + wy**2 + wz**2)
    I3 = sp.eye(3)
    theta = wn*s
    # R(s) = exp(-W s)
    K = W/wn
    return sp.simplify(I3 - sp.sin(theta)*K + (1 - sp.cos(theta))*(K*K))

def att_bias_phi_qd_isotropic(h, wx, wy, wz, qg, qbg):
    """
    State [dθ | b_g], with:
      A = -[ω]_x,  B = -I3,  D = 0.
      Qg = qg I3,  Qbg = qbg I3 (isotropic).
    Derivation uses:
      Phi_TL(h) = R(h) = exp(-W h)
      Phi_TR(h) = ∫_0^h e^{As} B ds = -∫_0^h R(s) ds
      Q_tt = ∫_0^h R(s) Qg R(s)^T ds + ∫_0^h B(s) Qbg B(s)^T ds
            = qg h I + qbg ∫_0^h B(s) B(s)^T ds
      Q_tb = (∫_0^h B(s) ds) qbg,   Q_bb = qbg h I
    We represent B(s) in the basis {I, K, K^2} with K=W/||ω|| and integrate only scalar coefficients.
    """
    s = sp.symbols('s', positive=True, real=True)
    W  = skew(wx, wy, wz)
    wn = sp.sqrt(wx**2 + wy**2 + wz**2)
    I3 = sp.eye(3)
    K  = W/wn

    # Rotation R(s)
    R = rodrigues_R(wx, wy, wz, s)        # 3x3

    # Phi blocks
    Phi_TL = rodrigues_R(wx, wy, wz, h)
    # Phi_TR = ∫_0^h e^{A s} (-I) ds = -∫_0^h R(s) ds
    Phi_TR = -sp.Matrix(3,3, lambda i,j: sp.integrate(R[i,j], (s,0,h)))
    Phi_BR = I3

    Phi = sp.Matrix.zeros(6,6)
    Phi[0:3,0:3] = sp.simplify(Phi_TL)
    Phi[0:3,3:6] = sp.simplify(Phi_TR)
    Phi[3:6,3:6] = I3

    # B(s) = -∫_0^s R(τ) dτ  = α0(s) I + α1(s) K + α2(s) K^2
    # Derive α_i(s) by integrating R(τ) in the form I - sin(θ)K + (1-cos θ)K^2:
    τ = sp.symbols('tau', positive=True, real=True)
    θ = wn*τ
    # Coefficients of R(τ) in I, K, K^2:
    cI  = 1
    cK  = -sp.sin(θ)
    cK2 = (1 - sp.cos(θ))
    # Integrate scalars to get coefficients of ∫R(τ)dτ:
    I_int  = sp.integrate(cI,  (τ,0,s))           # = s
    K_int  = sp.integrate(cK,  (τ,0,s))           # = (cos(wn s) - 1)/wn
    K2_int = sp.integrate(cK2, (τ,0,s))           # = s - sin(wn s)/wn
    # Hence: ∫_0^s R(τ)dτ = I_int*I + K_int*K + K2_int*K^2
    # So B(s) = - that:
    alpha0 = -I_int
    alpha1 = -K_int
    alpha2 = -K2_int

    # Now express B(s) and its transpose via the basis:
    # B(s) = α0 I + α1 K + α2 K^2
    # Note: K^T = -K, (K^2)^T = K^2, and K^3 = -K (since K^2 = -(I - u u^T) with unit u)
    # For products, we only need: I*I=I, K*K^T = (-K)*K = -K^2, K^2 * K^2 = K^4 = K^2*K^2.
    # We will reduce products to a linear combo of I and K^2 (the symmetric subspace).
    # First build symbolic B(s) in matrix form:
    B_sym = alpha0*I3 + alpha1*K + alpha2*(K*K)

    # IB = ∫_0^h B(s) ds (matrix, used in Q_tb)
    IB = sp.Matrix(3,3, lambda i,j: sp.integrate(B_sym[i,j], (s,0,h)))

    # Q components
    Qtt_gyro = qg * h * I3     # because R(s) I R(s)^T = I

    # For the bias term: ∫_0^h B(s)B(s)^T ds
    # Instead of expanding entries, integrate basis-wise by taking the matrix-valued integrand:
    BBt = sp.simplify(B_sym * B_sym.T)
    Qtt_bias = sp.Matrix(3,3, lambda i,j: sp.integrate(BBt[i,j], (s,0,h))) * qbg

    Qtt = sp.simplify(Qtt_gyro + Qtt_bias)
    Qbb = sp.simplify(qbg * h * I3)
    Qtb = sp.simplify(IB * qbg)

    Qd = sp.Matrix.zeros(6,6)
    Qd[0:3,0:3] = Qtt
    Qd[0:3,3:6] = Qtb
    Qd[3:6,0:3] = Qtb.T
    Qd[3:6,3:6] = Qbb
    Qd = sp.simplify((Qd + Qd.T)/2)

    return sp.simplify(Phi), sp.simplify(Qd)

# ---------- main demo ----------
if __name__ == "__main__":
    # Symbols
    h, tau = sp.symbols('h tau', positive=True)
    sigma2 = sp.symbols('sigma2', positive=True)  # stationary variance of a (OU)
    qg, qbg = sp.symbols('q_g q_bg', positive=True)
    wx, wy, wz = sp.symbols('w_x w_y w_z', real=True)

    # --- OU axis derivation ---
    Phi_ou, Qd_ou = ou_axis_phi_qd(h, tau, sigma2)
    pp("Phi_axis(h)  [OU axis v,p,S|a]", Phi_ou)
    pp("Qd_axis(h)   [OU axis v,p,S|a]", Qd_ou)
    to_latex_block("Phi_axis(h) latex", Phi_ou)
    to_latex_block("Qd_axis(h) latex",  Qd_ou)

    # --- Attitude+bias derivation (isotropic) ---
    Phi_ab, Qd_ab = att_bias_phi_qd_isotropic(h, wx, wy, wz, qg, qbg)
    pp("Phi_att+bias(h)  [dθ|b_g]", Phi_ab)
    pp("Qd_att+bias(h)   [dθ|b_g, isotropic]", Qd_ab)
    to_latex_block("Phi_att+bias(h) latex", Phi_ab)
    to_latex_block("Qd_att+bias(h) latex",  Qd_ab)
