# block-phi-qd-sym.py
# Build Phi(h) and Qd(h) from block-upper F=[[A,B],[0,D]] and L=[[L_top],[L_bot]].
# Uses symbolic integrals and exploits block structure:
#   Phi(h) = [[exp(Ah),  J(h)],[0, exp(Dh)]], J(h)=∫_0^h exp(As) B exp(Ds) ds
#   C(t)   = Phi(t)L = [exp(At)L_top + J(t)L_bot; exp(Dt)L_bot]
#   Qd(h)  = ∫_0^h C(t) Qc C(t)^T dt
#
# Includes fast specializations for:
#   - OU axis [v,p,S | a]
#   - Attitude + gyro-bias (Rodrigues + closed-form B(t))

import sympy as sp

# -------------------------
# Pretty helpers
# -------------------------
def pprint_header(s):
    print("\n" + s)
    print("-" * len(s))

def pprint_matrix(name, M):
    pprint_header(name)
    sp.pprint(sp.simplify(M), wrap_line=False, num_columns=140)

# -------------------------
# Core block tools
# -------------------------
def phi_blocks_from_F_blocks(A, B, D, h, t=None, expA=None, expD=None):
    """
    Build the block entries of Phi(h) for F=[[A,B],[0,D]].
    expA(t): callable returning exp(A t) (symbolic), required.
    expD(t): callable returning exp(D t) (symbolic), required.
    Returns: (Phi_top_left, Phi_top_right, Phi_bot_right)
    """
    if t is None:
        t = sp.symbols('t', positive=True, real=True)
    if expA is None or expD is None:
        raise ValueError("Provide expA(t) and expD(t) callables for speed/closed forms.")

    # exp(A t), exp(D t)
    eAt = expA(t)
    eDt = expD(t)

    # J(t) = ∫_0^t exp(A s) B exp(D s) ds
    s = sp.symbols('s', positive=True, real=True)
    integrand = expA(s) * B * expD(s)
    Jt = sp.Matrix(integrand.shape[0], integrand.shape[1],
                   lambda i,j: sp.integrate(integrand[i,j], (s, 0, t)))

    # Evaluate at h
    Phi_TL = sp.simplify(eAt.subs(t, h))
    Phi_BR = sp.simplify(eDt.subs(t, h))
    Phi_TR = sp.simplify(Jt.subs(t, h))
    return Phi_TL, Phi_TR, Phi_BR

def qd_from_blocks(A, B, D, L_top, L_bot, Qc, h, t=None, expA=None, expD=None):
    """
    Compute Qd(h) = ∫_0^h C(t) Qc C(t)^T dt using only columns C(t) = Phi(t) L.
    expA, expD are callables as in phi_blocks_from_F_blocks.
    """
    if t is None:
        t = sp.symbols('t', positive=True, real=True)
    if expA is None or expD is None:
        raise ValueError("Provide expA(t) and expD(t).")

    # exp(A t), exp(D t)
    eAt = expA(t)
    eDt = expD(t)

    # J(t)
    s = sp.symbols('s', positive=True, real=True)
    integrand_J = expA(s) * B * expD(s)
    Jt = sp.Matrix(integrand_J.shape[0], integrand_J.shape[1],
                   lambda i,j: sp.integrate(integrand_J[i,j], (s, 0, t)))

    # Columns C(t) = [eAt L_top + J(t) L_bot; eDt L_bot]
    C_top = sp.simplify(eAt * L_top + Jt * L_bot)
    C_bot = sp.simplify(eDt * L_bot)
    C = sp.Matrix.vstack(C_top, C_bot)  # (n+m) x r

    # Integrand: C(t) Qc C(t)^T
    integrand_Q = sp.simplify(C * Qc * C.T)

    # Integrate elementwise in t
    Qd = sp.Matrix(integrand_Q.shape[0], integrand_Q.shape[1],
                   lambda i,j: sp.integrate(integrand_Q[i,j], (t, 0, h)))
    # Symmetrize for hygiene
    Qd = sp.simplify((Qd + Qd.T) / 2)
    return Qd

# -------------------------
# Specializations (fast expA/expD)
# -------------------------
def expA_chain3_builder():
    """
    exp(A t) for A = [[0,0,0],[1,0,0],[0,1,0]] (nilpotent chain), exact polynomial.
    """
    A = sp.Matrix([[0,0,0],
                   [1,0,0],
                   [0,1,0]])
    I = sp.eye(3)
    A2 = A*A
    def expA(t):
        return I + A*t + A2*(t**2/2)  # since A^3 = 0
    return expA

def expD_scalar_builder(lambda_scalar):
    """
    exp(D t) for scalar D=[lambda].
    """
    def expD(t):
        return sp.Matrix([[sp.exp(lambda_scalar * t)]])
    return expD

def rodrigues_expA_builder(wx, wy, wz):
    """
    exp(A t) with A = -[w]_x using Rodrigues closed form.
    """
    W  = sp.Matrix([[0, -wz,  wy],
                    [wz,  0, -wx],
                    [-wy, wx,  0]])
    wn = sp.sqrt(wx**2 + wy**2 + wz**2)
    I3 = sp.eye(3)
    def expA(t):
        theta = wn*t
        K = W/wn
        return sp.simplify(I3 - sp.sin(theta)*K + (1 - sp.cos(theta))*(K*K))
    return expA

def expD_zero_builder(dim):
    """
    exp(D t) with D = 0_(dim x dim) ⇒ I.
    """
    I = sp.eye(dim)
    def expD(t):
        return I
    return expD

# -------------------------
# OU axis builder
# -------------------------
def build_OU_axis_blocks(tau):
    """
    State [v, p, S, a], F = [[A,B],[0,D]], L selects only 'a'.
      A = [[0,0,0],[1,0,0],[0,1,0]]
      B = [[1],[0],[0]]
      D = [[-1/tau]]
      L_top = 0_(3 x 1), L_bot = 1_(1 x 1)
    """
    A = sp.Matrix([[0,0,0],
                   [1,0,0],
                   [0,1,0]])
    B = sp.Matrix([[1],[0],[0]])
    D = sp.Matrix([[-1/tau]])
    L_top = sp.zeros(3,1)
    L_bot = sp.ones(1,1)  # [1]
    return A, B, D, L_top, L_bot

# -------------------------
# Attitude + gyro-bias builder
# -------------------------
def build_att_bias_blocks(wx, wy, wz):
    """
    State [δθ, b_g], F = [[-W, -I],[0, 0]], L injects gyro white (top) and bias RW (bottom).
      A = -W(ω), B = -I3, D = 0_(3), L_top = I3 (gyro noise), L_bot = I3 (bias RW)
    """
    W = sp.Matrix([[0, -wz,  wy],
                   [wz,  0, -wx],
                   [-wy, wx,  0]])
    A = -W
    B = -sp.eye(3)
    D = sp.zeros(3,3)
    L_top = sp.eye(3)
    L_bot = sp.eye(3)
    return A, B, D, L_top, L_bot

# -------------------------
# Demo / Usage
# -------------------------
if __name__ == "__main__":
    # Symbols
    h, tau = sp.symbols('h tau', positive=True)
    wx, wy, wz = sp.symbols('w_x w_y w_z', real=True)

    # Noise covariances (symbolic)
    sigma_x = sp.symbols('sigma_x', positive=True)
    qg, qbg = sp.symbols('q_g q_bg', positive=True)  # gyro white, bias RW

    # ------------------ OU axis example ------------------
    A_ou, B_ou, D_ou, Ltop_ou, Lbot_ou = build_OU_axis_blocks(tau)
    expA_ou = expA_chain3_builder()             # polynomial exp(A t)
    expD_ou = expD_scalar_builder(-1/tau)       # scalar exp(-t/tau)

    Phi_TL, Phi_TR, Phi_BR = phi_blocks_from_F_blocks(A_ou, B_ou, D_ou, h,
                                                      expA=expA_ou, expD=expD_ou)
    # FIX: use sp.diag (block diag), not Matrix.spdiag
    Phi_axis = sp.diag(Phi_TL, Phi_BR)          # 3x3 and 1x1 -> 4x4
    Phi_axis[0:3, 3:4] = Phi_TR                 # fill the upper-right block

    # Qc for OU axis: scalar 2*sigma^2/tau on 'a' input (since L is 1x1)
    Qc_ou = sp.Matrix([[2*sigma_x**2 / tau]])

    Qd_axis = qd_from_blocks(A_ou, B_ou, D_ou, Ltop_ou, Lbot_ou, Qc_ou, h,
                             expA=expA_ou, expD=expD_ou)

    pprint_matrix("Phi_axis(h) from (A,B,D)", Phi_axis)
    pprint_matrix("Qd_axis(h) from L, Qc", Qd_axis)

    # ------------------ Attitude + gyro-bias example ------------------
    A_ab, B_ab, D_ab, Ltop_ab, Lbot_ab = build_att_bias_blocks(wx, wy, wz)
    expA_ab = rodrigues_expA_builder(wx, wy, wz)  # Rodrigues
    expD_ab = expD_zero_builder(3)                # zero 3x3 ⇒ I

    Phi_TL_ab, Phi_TR_ab, Phi_BR_ab = phi_blocks_from_F_blocks(A_ab, B_ab, D_ab, h,
                                                               expA=expA_ab, expD=expD_ab)
    Phi_ab = sp.Matrix.zeros(6,6)
    Phi_ab[0:3,0:3] = Phi_TL_ab
    Phi_ab[0:3,3:6] = Phi_TR_ab
    Phi_ab[3:6,3:6] = Phi_BR_ab

    # Qc for attitude+bias: blockdiag(qg*I3, qbg*I3)
    Qc_ab = sp.diag(qg, qg, qg, qbg, qbg, qbg)

    Qd_ab = qd_from_blocks(A_ab, B_ab, D_ab, Ltop_ab, Lbot_ab, Qc_ab, h,
                           expA=expA_ab, expD=expD_ab)

    pprint_matrix("Phi_att+bias(h) from (A,B,D)", Phi_ab)
    pprint_matrix("Qd_att+bias(h) from L, Qc", Qd_ab)
