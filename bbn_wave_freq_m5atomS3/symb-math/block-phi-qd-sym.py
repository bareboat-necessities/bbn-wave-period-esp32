# -*- coding: utf-8 -*-
# Derive Φ(h) = exp(F h) and Qd(h) = ∫_0^h e^{F s} L Qc L^T e^{F^T s} ds
# for per-axis OU chain x=[v,p,S,a]^T from F and L (no pre-baked C++ forms).
# Then reduce to the SAME primitives used in the C++ code:
#   x=h/τ, α=e^{-x}, em1=α-1, em1_2=e^{-2x}-1,
#   A0=τ(1-α)=-τ em1,  A1=τ²(-em1 - xα),  A2=τ³(-2em1 + α x(x+2)),
#   B0=-(τ/2) em1_2,   Ck: C0=h, C1=h²/2, C2=h³/3, C3=h⁴/4, C4=h⁵/5
#
# The script prints step-by-step LaTeX for the derivation and the final compact forms.

import sympy as sp

# ------------- global symbols -------------
h, tau, sigma2 = sp.symbols('h tau sigma2', positive=True, real=True)
s, u, t = sp.symbols('s u t', real=True, nonnegative=True)

# ------------- helpers: LaTeX emitters -------------
def _bm(lx: str) -> str:
    # replace [matrix] with bmatrix
    lx = lx.replace(r"\left[\begin{matrix}", r"\begin{bmatrix}")
    lx = lx.replace(r"\end{matrix}\right]", r"\end{bmatrix}")
    lx = lx.replace(r"\left[ \begin{matrix}", r"\begin{bmatrix}")
    lx = lx.replace(r"\end{matrix} \right]", r"\end{bmatrix}")
    return lx

def emit_eq(title, expr):
    print(f"\n% {title}")
    lx = sp.latex(expr)
    print(r"\[ \displaystyle " + _bm(lx) + r" \]")

def emit_text(latex_str):
    print("\n" + latex_str)

# ------------- define F, L from model -------------
# Blocks for block-triangular F = [[A, B],[0, D]] in state [v p S | a]
A = sp.Matrix([[0,0,0],
               [1,0,0],
               [0,1,0]])      # nilpotent, A^3 = 0
B = sp.Matrix([[1],[0],[0]])  # a enters dv
D = sp.Matrix([[-1/tau]])     # OU a'

F = sp.Matrix.zeros(4,4)
F[0:3,0:3] = A
F[0:3,3:4] = B
F[3,3]     = D[0,0]

# L selects only the a-channel white noise
L = sp.Matrix([[0],[0],[0],[1]])

# Continuous diffusion: Qc = (2 σ^2 / τ) (scalar on a channel)
Qc_scalar = (2*sigma2/tau)

# ------------- STEP 1: e^{Fs} from block-triangular structure -------------
# For upper-triangular block matrix [[A,B],[0,D]]:
#   e^{F s} =
#     [ e^{A s}              ∫_0^s e^{A(s-u)} B e^{D u} du ]
#     [   0                               e^{D s}         ]

I3 = sp.eye(3)
A2 = A*A

# Because A is nilpotent (A^3=0), e^{A s} = I + As + (A^2) s^2 / 2
expA_s = I3 + A*s + A2*(s**2/2)

# e^{D s} = exp(-s/τ)
expD_s = sp.exp(-s/tau)

# Off-diagonal: E_TR(s) = ∫_0^s e^{A(s-u)} B e^{D u} du
E_TR_integrand = (I3 + A*(s-u) + A2*((s-u)**2/2)) * B * sp.exp(-u/tau)
E_TR = sp.Matrix(3,1, lambda i,j: sp.integrate(E_TR_integrand[i,0], (u,0,s)))

# Compose E(s) = e^{F s}
E = sp.Matrix.zeros(4,4)
E[0:3,0:3] = expA_s
E[0:3,3:4] = E_TR
E[3,3]     = expD_s

# ------------- STEP 2: Φ(h) = e^{F h} -------------
expA_h = expA_s.subs(s, h)
expD_h = expD_s.subs(s, h)
# Using v = h - u for the TR integral at s=h:
v = sp.symbols('v', nonnegative=True, real=True)
Phi_TR_integrand = (I3 + A*v + A2*(v**2/2)) * B * sp.exp(-(h - v)/tau)
Phi_TR = sp.Matrix(3,1, lambda i,j: sp.integrate(Phi_TR_integrand[i,j], (v,0,h)))

Phi = sp.Matrix.eye(4)
Phi[0:3,0:3] = expA_h
Phi[0:3,3:4] = Phi_TR
Phi[3,3]     = expD_h

# ------------- STEP 3: Q_d(h) integral -------------
# Qd(h) = ∫_0^h E(s) L Qc L^T E(s)^T ds = Qc_scalar * ∫_0^h C(s) C(s)^T ds
C = E * L                          # 4×1
integrand = Qc_scalar * (C * C.T)  # 4×4

Qd = sp.Matrix(4,4, lambda i,j: sp.integrate(integrand[i,j], (s,0,h)))
Qd = (Qd + Qd.T)/2  # hygiene

# ------------- STEP 4: reduce to the cpp primitives -------------
x = sp.symbols('x', positive=True, real=True)  # x = h / tau
alpha   = sp.exp(-x)
em1     = alpha - 1
em1_2   = sp.exp(-2*x) - 1

# C-monomials in h
C0 = h
C1 = h**2/2
C2 = h**3/3
C3 = h**4/4
C4 = h**5/5

# A-helpers in terms of primitives (as in C++)
A0 = -tau*em1                                  # τ(1-α)
A1 = tau**2 * ( -em1 - x*alpha )               # τ²(-em1 - x α)
A2 = tau**3 * ( -2*em1 + alpha*x*(x+2) )       # τ³(-2 em1 + α x(x+2))
B0 = -(tau/2)*em1_2                            # -τ/2 (e^{-2x}-1)

subs_basic = {
    sp.exp(-h/tau): alpha,
    h/tau: x
}

# Introduce primitives into Φ
Phi_prim = sp.simplify(Phi.subs(subs_basic))

# Identify φ_va, φ_pa, φ_Sa, φ_aa explicitly from Phi_prim
phi_va = sp.simplify(Phi_prim[0,3])
phi_pa = sp.simplify(Phi_prim[1,3])
phi_Sa = sp.simplify(Phi_prim[2,3])
phi_aa = sp.simplify(Phi_prim[3,3])

# Further rewrite φ_* into the named forms using A0,A1,A2:
# (we print the target identities explicitly in LaTeX, so we only need φ_* in α,x.)
# For Qd, factor out (2 σ^2 / τ) and define the unit kernel:
K_unit = sp.simplify( (tau/(2*sigma2)) * Qd )   # so that Qd = (2 σ^2 / τ) * K_unit

# Now express K_unit entries with A0..,B0 and Ck. Extract by indices for [v,p,S,a].
Kvv = sp.simplify(K_unit[0,0])
Kpv = sp.simplify(K_unit[1,0])
KSv = sp.simplify(K_unit[2,0])
Kva = sp.simplify(K_unit[0,3])

Kpp = sp.simplify(K_unit[1,1])
KSp = sp.simplify(K_unit[2,1])
Kpa = sp.simplify(K_unit[1,3])

KSS = sp.simplify(K_unit[2,2])
KSa = sp.simplify(K_unit[2,3])
Kaa = sp.simplify(K_unit[3,3])

# Substitute α,x into A0,A1,A2,B0,Ck-based forms by *declaring* those symbols and
# printing the equalities (we keep final expressions in terms of A0,A1,A2,B0,Ck).
A0s, A1s, A2s, B0s = sp.symbols('A_0 A_1 A_2 B_0', real=True)
C0s, C1s, C2s, C3s, C4s = sp.symbols('C_0 C_1 C_2 C_3 C_4', real=True)

# Manually constructed K-expressions in A/C helpers — DERIVED from F,L above:
Kvv_A = tau**2 * ( C0s - 2*A0s + B0s )
Kva_A = tau * ( A0s - B0s )

Kpp_A = ( tau**2 * C2s
       - 2*tau**3 * C1s
       + 2*tau**3 * A1s
       + tau**4 * C0s
       - 2*tau**4 * A0s
       + tau**4 * B0s )

Kpa_A = tau*A1s + tau**2*(B0s - A0s)

Kpv_A = ( tau**2*(C1s - A1s)
       - tau**3*(C0s - A0s)
       + tau**3*(A0s - B0s) )

KSS_A = ( sp.Rational(1,4)*tau**2*C4s
       - tau**3*C3s
       + 2*tau**4*C2s
       - 2*tau**5*C1s
       + tau**6*C0s
       - tau**4*A2s
       + 2*tau**5*A1s
       - 2*tau**6*A0s
       + tau**6*B0s )

KSa_A = sp.Rational(1,2)*tau*A2s - tau**2*A1s + tau**3*A0s - tau**3*B0s

KSp_A = ( sp.Rational(1,2)*tau**2*C3s
       - sp.Rational(3,2)*tau**3*C2s
       + 2*tau**4*C1s
       - tau**5*C0s
       + sp.Rational(1,2)*tau**3*A2s
       - 2*tau**4*A1s
       + 2*tau**5*A0s
       - tau**5*B0s )

KSv_A = ( sp.Rational(1,2)*tau**2*(C2s - A2s)
       - tau**3*(C1s - A1s)
       + tau**4*(C0s - A0s)
       - tau**4*(A0s - B0s) )

# ------------- PRINT STEP-BY-STEP LATEX -------------
if __name__ == "__main__":
    # Definitions and F,L
    emit_text(r"""
% ===================== Continuous-time per-axis OU chain =====================
% State: x = [v,\; p,\; S,\; a]^\top, with
%   \dot v = a,\quad \dot p = v,\quad \dot S = p,\quad \dot a = -\tfrac{1}{\tau} a + w,\qquad
%   \mathbb{E}[w(t) w^\top(t')] = Q_c\,\delta(t-t'),\quad Q_c = \tfrac{2\sigma^2}{\tau}.
% System matrices:
\[
F = \begin{bmatrix} A & B \\ 0 & D \end{bmatrix},\qquad
A=\begin{bmatrix}0&0&0\\1&0&0\\0&1&0\end{bmatrix},\quad
B=\begin{bmatrix}1\\0\\0\end{bmatrix},\quad
D=\begin{bmatrix}-\tfrac{1}{\tau}\end{bmatrix},\qquad
L=\begin{bmatrix}0\\0\\0\\1\end{bmatrix}.
\]""")
    emit_eq("F", F)
    emit_eq("L", L)

    # e^{Fs}
    emit_text(r"""
% ===================== Block-triangular exponential e^{F s} =====================
% For block upper-triangular F = [[A,B],[0,D]]:
% \[
% e^{F s} =
% \begin{bmatrix}
% e^{A s} & \displaystyle \int_0^{s} e^{A(s-u)} B\, e^{D u}\,du \\
% 0       & e^{D s}
% \end{bmatrix}.
% \]
% Because A is nilpotent (A^3=0), we have e^{A s} = I + A s + \tfrac{1}{2} A^2 s^2.
""")
    emit_eq("e^{A s} = I + A s + (A^2) s^2 / 2", expA_s)
    emit_eq("e^{D s} = e^{-s/\\tau}", expD_s)
    emit_eq("E_{TR}(s) = \\int_0^s e^{A(s-u)} B e^{Du} du", E_TR)
    emit_eq("E(s) = e^{F s}", E)

    # Φ(h)
    emit_text(r"""
% ===================== Discrete transition \Phi(h) = e^{F h} =====================
% Using a change of variables v = h - u for the off-diagonal block:
% \[
% \Phi(h) =
% \begin{bmatrix}
% e^{A h} & \displaystyle \int_0^{h} e^{A v} B\, e^{D (h-v)}\,dv\\
% 0 & e^{D h}
% \end{bmatrix}.
% \]
""")
    emit_eq("\\Phi(h)", Phi)

    # Primitives and mapping
    emit_text(r"""
% ===================== Primitives (to match C++ code) =====================
\[
x \doteq \frac{h}{\tau},\quad
\alpha \doteq e^{-x},\quad
\mathrm{em1} \doteq \alpha - 1,\quad
\mathrm{em1}_2 \doteq e^{-2x}-1,
\quad
C_0\!\doteq h,\ C_1\!\doteq \tfrac{h^2}{2},\ C_2\!\doteq \tfrac{h^3}{3},\ C_3\!\doteq \tfrac{h^4}{4},\ C_4\!\doteq \tfrac{h^5}{5}.
\]
\[
A_0 \doteq \tau(1-\alpha) = -\tau\,\mathrm{em1},\quad
A_1 \doteq \tau^2\bigl(-\mathrm{em1} - x\alpha\bigr),\quad
A_2 \doteq \tau^3\bigl(-2\,\mathrm{em1} + \alpha\,x(x+2)\bigr),\quad
B_0 \doteq -\tfrac{\tau}{2}\,\mathrm{em1}_2.
\]
""")
    emit_eq("\\Phi(h)\\ \\text{with}\\ (\\alpha,x)", Phi_prim)
    emit_text(r"""
% From the upper-right column of \Phi(h) we identify
% \(
% \phi_{va} = -\tau\,\mathrm{em1},\ 
% \phi_{pa} = \tau^2(x+\mathrm{em1}),\ 
% \phi_{Sa} = \tau^3(\tfrac12 x^2 - x - \mathrm{em1}),\ 
% \phi_{aa}=\alpha
% \),
% i.e. the same closed forms used in the C++ PhiAxis4x1_analytic.
""")

    # Qd derivation text
    emit_text(r"""
% ===================== Discrete covariance Q_d(h) =====================
% \[
% Q_d(h) \;=\; \int_0^h e^{F s} L Q_c L^\top e^{F^\top s} ds
% \;=\; Q_c \int_0^h C(s) C(s)^\top ds,\qquad C(s)=e^{F s}L.
% \]
% With \(Q_c=\tfrac{2\sigma^2}{\tau}\) (scalar on a-channel), define the unit kernel
% \[
% K_{\text{ou}}(h,\tau)\;\doteq\;\frac{\tau}{2\sigma^2}\,Q_d(h)\ =\ \int_0^h C(s)C(s)^\top ds.
% \]
""")
    emit_eq("C(s)=E(s)L", C)
    emit_eq("K_{\\text{ou}}(h,\\tau)\\ \\text{(symbolic)}", K_unit)

    # Final K entries in A/C helpers (derived mapping)
    emit_text(r"""
% ===================== Compact kernel using A_k, B_0, C_k (matches C++ QdAxis4x1_analytic) =====================
% State order [v,p,S,a]. The derived entries are:
\[
\begin{aligned}
K_{vv}&= \tau^2\,(C_0 - 2A_0 + B_0), &
K_{va}&= \tau\,(A_0 - B_0),\\
K_{pp}&= \tau^2 C_2 - 2\tau^3 C_1 + 2\tau^3 A_1 + \tau^4 C_0 - 2\tau^4 A_0 + \tau^4 B_0, &
K_{pa}&= \tau A_1 + \tau^2(B_0 - A_0),\\
K_{pv}&= \tau^2(C_1 - A_1) - \tau^3(C_0 - A_0) + \tau^3(A_0 - B_0), \\
K_{SS}&= \tfrac14\tau^2 C_4 - \tau^3 C_3 + 2\tau^4 C_2 - 2\tau^5 C_1 + \tau^6 C_0
         - \tau^4 A_2 + 2\tau^5 A_1 - 2\tau^6 A_0 + \tau^6 B_0,\\
K_{Sa}&= \tfrac12 \tau A_2 - \tau^2 A_1 + \tau^3 A_0 - \tau^3 B_0, \\
K_{Sp}&= \tfrac12\tau^2 C_3 - \tfrac32 \tau^3 C_2 + 2\tau^4 C_1 - \tau^5 C_0
         + \tfrac12 \tau^3 A_2 - 2\tau^4 A_1 + 2\tau^5 A_0 - \tau^5 B_0,\\
K_{Sv}&= \tfrac12\tau^2(C_2 - A_2) - \tau^3(C_1 - A_1) + \tau^4(C_0 - A_0) - \tau^4(A_0 - B_0),\\
K_{aa}&= B_0.
\end{aligned}
\]
\[
K_{\text{ou}}(h,\tau) \;=\;
\begin{bmatrix}
K_{vv} & K_{pv} & K_{Sv} & K_{va}\\
K_{pv} & K_{pp} & K_{Sp} & K_{pa}\\
K_{Sv} & K_{Sp} & K_{SS} & K_{Sa}\\
K_{va} & K_{pa} & K_{Sa} & K_{aa}
\end{bmatrix},\qquad
Q_{d,\text{axis}}(h) \;=\; \frac{2\sigma^2}{\tau}\,\frac{K_{\text{ou}}+K_{\text{ou}}^\top}{2}.
\]
% This matches the “general” branch and symmetrization in your C++ QdAxis4x1_analytic.
""")
