# block-phi-qd-sym.py
import sympy as sp

# ---------- pretty ----------
def pp(name, M):
    print("\n" + name + "\n" + "-"*len(name))
    sp.pprint(sp.simplify(M), wrap_line=False, num_columns=140)

# ---------- core: Phi blocks ----------
def phi_blocks_from_F_blocks(A, B, D, h, *, expA, expD, t=None):
    """
    For F=[[A,B],[0,D]], return Phi blocks:
      Phi_TL = exp(A h), Phi_TR = ∫_0^h exp(A s) B exp(D s) ds, Phi_BR = exp(D h)
    """
    if t is None:
        t = sp.symbols('t', positive=True, real=True)
    s = sp.symbols('s', positive=True, real=True)

    eAt = expA(t)
    eDt = expD(t)

    integrand = expA(s) * B * expD(s)
    Phi_TR_t = sp.Matrix(integrand.shape[0], integrand.shape[1],
                         lambda i,j: sp.integrate(integrand[i,j], (s, 0, t)))

    Phi_TL = sp.simplify(eAt.subs(t, h))
    Phi_TR = sp.simplify(Phi_TR_t.subs(t, h))
    Phi_BR = sp.simplify(eDt.subs(t, h))
    return Phi_TL, Phi_TR, Phi_BR

# ---------- core: Qd via column blocks ----------
def qd_from_blocks(A, B, D, L_top, L_bot, Qc, h, *, expA, expD, t=None):
    """
    Build Qd(h) = ∫_0^h C(t) Qc C(t)^T dt, with
      C(t) = [ e^{At} L_top | J(t) L_bot ;
               0            | e^{Dt} L_bot ]
    Handles empty-channel blocks (0 columns) automatically.
    """
    if t is None:
        t = sp.symbols('t', positive=True, real=True)
    s = sp.symbols('s', positive=True, real=True)

    eAt = expA(t)
    eDt = expD(t)

    # J(t) = ∫_0^t e^{A s} B e^{D s} ds
    integrand_J = expA(s) * B * expD(s)
    Jt = sp.Matrix(integrand_J.shape[0], integrand_J.shape[1],
                   lambda i,j: sp.integrate(integrand_J[i,j], (s, 0, t)))

    # Channel widths
    rg = L_top.shape[1]  # gyro (or first group) channels
    rb = L_bot.shape[1]  # OU/bias (or second group) channels

    blocks_top, blocks_bot = [], []

    if rg > 0:
        blocks_top.append(sp.simplify(eAt * L_top))                  # n x rg
        blocks_bot.append(sp.zeros(D.shape[0], rg))                   # m x rg

    if rb > 0:
        blocks_top.append(sp.simplify(Jt * L_bot))                    # n x rb
        blocks_bot.append(sp.simplify(eDt * L_bot))                   # m x rb

    if not blocks_top:
        raise ValueError("Both L_top and L_bot have zero columns; no inputs => Qd=0.")

    C_top = blocks_top[0] if len(blocks_top)==1 else sp.Matrix.hstack(*blocks_top)
    C_bot = blocks_bot[0] if len(blocks_bot)==1 else sp.Matrix.hstack(*blocks_bot)
    C = sp.Matrix.vstack(C_top, C_bot)  # (n+m) x r, r = rg+rb (skipping empties)

    # Ensure Qc has matching width r
    r = C.shape[1]
    if Qc.shape != (r, r):
        raise ValueError(f"Qc shape {Qc.shape} must match total input cols r={r}.")

    integrand_Q = sp.simplify(C * Qc * C.T)
    Qd = sp.Matrix(integrand_Q.shape[0], integrand_Q.shape[1],
                   lambda i,j: sp.integrate(integrand_Q[i,j], (t, 0, h)))
    return sp.simplify((Qd + Qd.T)/2)

# ---------- exp builders ----------
def expA_chain3_builder():
    # A = [[0,0,0],[1,0,0],[0,1,0]], nilpotent => exact polynomial
    A = sp.Matrix([[0,0,0],[1,0,0],[0,1,0]])
    I = sp.eye(3); A2 = A*A
    def expA(t): return I + A*t + A2*(t**2/2)
    return expA

def expD_scalar_builder(lam):
    def expD(t): return sp.Matrix([[sp.exp(lam*t)]])
    return expD

def rodrigues_expA_builder(wx, wy, wz):
    W  = sp.Matrix([[0, -wz,  wy],[wz, 0, -wx],[-wy, wx, 0]])
    wn = sp.sqrt(wx**2 + wy**2 + wz**2)
    I3 = sp.eye(3)
    def expA(t):
        theta = wn*t
        # Handle wn==0 by series; sympy will keep this symbolic, fine for closed form
        K = W/wn
        return sp.simplify(I3 - sp.sin(theta)*K + (1 - sp.cos(theta))*(K*K))
    return expA

def expD_zero_builder(dim):
    I = sp.eye(dim)
    def expD(t): return I
    return expD

# ---------- blocks for your subsystems ----------
def build_OU_axis_blocks(tau):
    """
    State [v,p,S | a]:
      A = [[0,0,0],[1,0,0],[0,1,0]]
      B = [[1],[0],[0]]
      D = [[-1/tau]]
      L_top = zeros(3,0)  (no 'gyro' channel here)
      L_bot = [[1]]       (the OU 'a' channel)
    """
    A = sp.Matrix([[0,0,0],[1,0,0],[0,1,0]])
    B = sp.Matrix([[1],[0],[0]])
    D = sp.Matrix([[-1/tau]])
    L_top = sp.zeros(3, 0)  # *** zero columns to avoid dummy channel ***
    L_bot = sp.ones(1, 1)
    return A, B, D, L_top, L_bot

def build_att_bias_blocks(wx, wy, wz):
    """
    State [δθ | b_g]:
      A = -[ω]_x, B = -I3, D = 0_3
      L_top = I3 (gyro white), L_bot = I3 (bias RW)
    """
    W = sp.Matrix([[0, -wz,  wy],[wz, 0, -wx],[-wy, wx, 0]])
    A = -W
    B = -sp.eye(3)
    D = sp.zeros(3,3)
    L_top = sp.eye(3)
    L_bot = sp.eye(3)
    return A, B, D, L_top, L_bot

# ---------- demo / usage ----------
if __name__ == "__main__":
    h, tau = sp.symbols('h tau', positive=True)
    wx, wy, wz = sp.symbols('w_x w_y w_z', real=True)
    sigma_a = sp.symbols('sigma_a', positive=True)
    qg, qbg = sp.symbols('q_g q_bg', positive=True)

    # === OU axis (single channel) ===
    A_ou,B_ou,D_ou,Lt_ou,Lb_ou = build_OU_axis_blocks(tau)
    expA_ou = expA_chain3_builder()
    expD_ou = expD_scalar_builder(-1/tau)

    Phi_TL, Phi_TR, Phi_BR = phi_blocks_from_F_blocks(A_ou,B_ou,D_ou,h,expA=expA_ou,expD=expD_ou)
    Phi_axis = sp.diag(Phi_TL, Phi_BR); Phi_axis[0:3, 3:4] = Phi_TR

    # Qc for OU axis: width r = cols(L_top)+cols(L_bot) = 0 + 1 = 1
    Qc_ou = sp.Matrix([[2*sigma_a**2 / tau]])
    Qd_axis = qd_from_blocks(A_ou,B_ou,D_ou,Lt_ou,Lb_ou,Qc_ou,h,expA=expA_ou,expD=expD_ou)

    pp("Phi_axis(h)", Phi_axis)
    pp("Qd_axis(h)",  Qd_axis)

    # === Attitude + bias (two 3-chan blocks) ===
    A_ab,B_ab,D_ab,Lt_ab,Lb_ab = build_att_bias_blocks(wx,wy,wz)
    expA_ab = rodrigues_expA_builder(wx,wy,wz)
    expD_ab = expD_zero_builder(3)

    Phi_TL_ab, Phi_TR_ab, Phi_BR_ab = phi_blocks_from_F_blocks(A_ab,B_ab,D_ab,h,expA=expA_ab,expD=expD_ab)
    Phi_ab = sp.Matrix.zeros(6,6)
    Phi_ab[0:3,0:3] = Phi_TL_ab
    Phi_ab[0:3,3:6] = Phi_TR_ab
    Phi_ab[3:6,3:6] = Phi_BR_ab

    # Qc width r = cols(L_top)+cols(L_bot) = 3+3 = 6
    Qc_ab = sp.diag(qg, qg, qg, qbg, qbg, qbg)
    Qd_ab = qd_from_blocks(A_ab,B_ab,D_ab,Lt_ab,Lb_ab,Qc_ab,h,expA=expA_ab,expD=expD_ab)

    pp("Phi_att+bias(h)", Phi_ab)
    pp("Qd_att+bias(h)",  Qd_ab)
