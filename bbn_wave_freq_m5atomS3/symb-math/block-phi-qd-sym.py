# block-phi-qd-sym.py
# Derives Phi(h) and Qd(h) symbolically from F and L, prints CLEAN LaTeX

import sympy as sp

# ---------- Utilities to emit clean LaTeX ----------
def _sanitize_matrix_brackets(lx: str) -> str:
    # Turn \left[\begin{matrix} ... \end{matrix}\right] into \begin{bmatrix} ... \end{bmatrix}
    lx = lx.replace(r"\left[\begin{matrix}", r"\begin{bmatrix}")
    lx = lx.replace(r"\end{matrix}\right]", r"\end{bmatrix}")
    # Also handle nested or variant spacing
    lx = lx.replace(r"\left[ \begin{matrix}", r"\begin{bmatrix}")
    lx = lx.replace(r"\end{matrix} \right]", r"\end{bmatrix}")
    return lx

def latex_block(title: str, expr) -> None:
    print("\n% " + title)
    lx = sp.latex(expr)
    lx = _sanitize_matrix_brackets(lx)
    print(r"\[ \displaystyle " + lx + r" \]")

def latex_text_block(text: str) -> None:
    # For paragraphs or aligned math already provided as LaTeX
    print(text)

# ============================================================
# (1) OU per-axis derivation directly from F and L
#     x = [v, p, S, a]^T
#     dv = a, dp = v, dS = p;  da = -(1/tau) a + w
#     F = [[A,B],[0,D]]   with A nilpotent, B = e_v, D = -1/tau
#     L selects only 'a' noise; Qc_unit = 2/tau  (unit stationary variance)
# ============================================================
def derive_ou_axis_phi_and_Kou():
    h, tau = sp.symbols('h tau', positive=True, real=True)
    s, u, v = sp.symbols('s u v', real=True, nonnegative=True)

    A = sp.Matrix([[0,0,0],
                   [1,0,0],
                   [0,1,0]])          # nilpotent (A^3=0)
    B = sp.Matrix([[1],[0],[0]])      # a enters dv
    D = sp.Matrix([[-1/tau]])         # OU a'

    I3 = sp.eye(3)
    A2 = A*A

    # e^{A h} = I + Ah + A^2 h^2/2
    Phi_TL = I3 + A*h + A2*(h**2/2)

    # TR(h) = ∫_0^h e^{A(h-u)} B e^{D u} du
    TR_integrand = (I3 + A*v + A2*(v**2/2)) * B * sp.exp(-(h - v)/tau)
    Phi_TR = sp.Matrix(3,1, lambda i,j: sp.integrate(TR_integrand[i,j], (v,0,h)))

    Phi_BR = sp.exp(-h/tau)

    Phi_axis = sp.eye(4)
    Phi_axis[0:3,0:3] = Phi_TL
    Phi_axis[0:3,3:4] = Phi_TR
    Phi_axis[3,3]     = Phi_BR

    # Discrete covariance (unit kernel K_ou): Qc_unit = 2/tau on 'a' channel
    TR_s_integrand = (I3 + A*(s-u) + A2*((s-u)**2/2)) * B * sp.exp(-u/tau)
    E_TR = sp.Matrix(3,1, lambda i,j: sp.integrate(TR_s_integrand[i,0], (u,0,s)))
    E_BR = sp.exp(-s/tau)

    C_top = E_TR                 # 3×1
    C_bot = sp.Matrix([[E_BR]])  # 1×1
    qc_unit = 2/tau

    integrand = sp.zeros(4,4)
    integrand[0:3,0:3] = qc_unit * (C_top * C_top.T)        # 3x3
    integrand[0:3,3:4] = qc_unit * (C_top * C_bot.T)        # 3x1
    integrand[3:4,0:3] = qc_unit * (C_bot * C_top.T)        # 1x3
    integrand[3,3]     = qc_unit * (C_bot * C_bot.T)[0,0]   # scalar

    Qd_unit = sp.Matrix(4,4, lambda i,j: sp.integrate(integrand[i,j], (s,0,h)))
    Qd_unit = (Qd_unit + Qd_unit.T)/2

    K_ou = sp.simplify(Qd_unit)
    return h, tau, Phi_axis, K_ou

# ============================================================
# (2) Kronecker 3-axis assembly (shared tau)
#     Phi_LL = I3 ⊗ Phi_axis
#     Q_LL   = Sigma_aw ⊗ K_ou(h,tau)
# ============================================================
def emit_kronecker_forms():
    h, tau, Phi_axis, K_ou = derive_ou_axis_phi_and_Kou()

    sxx, sxy, sxz, syy, syz, szz = sp.symbols('s_{xx} s_{xy} s_{xz} s_{yy} s_{yz} s_{zz}', real=True)
    Sigma_aw = sp.Matrix([[sxx, sxy, sxz],
                          [sxy, syy, syz],
                          [sxz, syz, szz]])
    I3 = sp.eye(3)

    Phi_LL = sp.kronecker_product(I3, Phi_axis)   # 12×12
    Q_LL   = sp.kronecker_product(Sigma_aw, K_ou) # 12×12

    latex_text_block(r"""
%======================== Kronecker Assembly (shared \tau) ========================
% Linear 3-axis OU subsystem (axes uncoupled in F, correlated in noise via \Sigma_{aw})
\[
\Phi_{LL}(h) \;=\; I_3 \otimes \Phi_{\text{axis}}(h), 
\qquad
Q_{LL}(h) \;=\; \Sigma_{aw} \;\otimes\; K_{\text{ou}}(h,\tau),
\]
where \(\Phi_{\text{axis}}(h)\in\mathbb{R}^{4\times 4}\) is the per-axis transition for \([v,p,S,a]\),
and \(K_{\text{ou}}(h,\tau)\in\mathbb{R}^{4\times 4}\) is the axis-unit discrete covariance kernel
obtained from \(Q_c=(2/\tau)\) on the \(a\)-channel (unit stationary variance), independent of \(\Sigma_{aw}\).
""")
    latex_block(r"\Phi_{\text{axis}}(h)", Phi_axis)
    latex_block(r"K_{\text{ou}}(h,\tau)", K_ou)

# ============================================================
# (3) Attitude + gyro-bias block [δθ | b_g]
#     F = [[A,B],[0,0]],  A = -[ω]_×, B = -I, D = 0
# ============================================================
def derive_att_bias_to_latex():
    h = sp.symbols('h', positive=True, real=True)
    wx, wy, wz = sp.symbols('w_x w_y w_z', real=True)

    W = sp.Matrix([[0, -wz,  wy],
                   [wz,  0, -wx],
                   [-wy, wx,  0]])
    A = -W
    I3 = sp.eye(3)
    B = -I3

    wn = sp.sqrt(wx**2 + wy**2 + wz**2)
    K = W/wn

    Phi_TL = I3 - sp.sin(wn*h)*K + (1 - sp.cos(wn*h))*(K*K)

    t = sp.symbols('t', positive=True, real=True)
    Rt = I3 - sp.sin(wn*t)*K + (1 - sp.cos(wn*t))*(K*K)
    Phi_TR = -sp.Matrix(3,3, lambda i,j: sp.integrate(Rt[i,j], (t,0,h)))

    Phi = sp.Matrix.zeros(6,6)
    Phi[0:3,0:3] = Phi_TL
    Phi[0:3,3:6] = Phi_TR
    Phi[3:6,3:6] = I3

    latex_block("Attitude+Bias: \\Phi(h)=e^{F h} from F = \\begin{bmatrix}A&B\\\\0&0\\end{bmatrix},\\ A=-[\\omega]_\\times,\\ B=-I", Phi)

    latex_text_block(r"""
% Q_d(h) from F and L (isotropic continuous covariances):
%   Q_g = q_g I_3  (gyro),  Q_{bg} = q_{bg} I_3  (bias)
%   L arranged so gyro noise drives attitude block, bias noise drives bias block.
\[
\begin{aligned}
R(s) &= e^{A s} \;=\; I - \sin(\omega_n s)\,K + \bigl(1-\cos(\omega_n s)\bigr)K^2,\\[0.25em]
B(s) &= \int_0^s e^{A u} B\,du \;=\; -\!\int_0^s R(u)\,du
     \;=\; \alpha_0(s)\,I + \alpha_1(s)\,K + \alpha_2(s)\,K^2,\\
\alpha_0(s) &= -\,s,\qquad
\alpha_1(s) \;=\; -\frac{\cos(\omega_n s)-1}{\omega_n},\qquad
\alpha_2(s) \;=\; -\Bigl(s - \frac{\sin(\omega_n s)}{\omega_n}\Bigr),\\[0.4em]
Q_{tt}(h) &= \int_0^h R(s)\,q_g I\,R(s)^\top ds \;+\; \int_0^h B(s)\,q_{bg} I\,B(s)^\top ds
          \;=\; q_g\,h\,I \;+\; q_{bg}\!\int_0^h B(s)B(s)^\top ds,\\
Q_{tb}(h) &= \Bigl(\int_0^h B(s)\,ds\Bigr)\,q_{bg},\qquad
Q_{bb}(h) \;=\; q_{bg}\,h\,I,
\end{aligned}
\]
where \( \omega_n=\|\boldsymbol{\omega}\|,\ K=[\boldsymbol{\omega}]_\times/\omega_n\).
% Expand these integrals in the appendix if you want explicit sin/cos closed forms.
""")

# -------------------- main --------------------
if __name__ == "__main__":
    emit_kronecker_forms()
    derive_att_bias_to_latex()
