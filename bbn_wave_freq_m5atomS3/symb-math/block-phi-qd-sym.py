# -*- coding: utf-8 -*-
# Derive Φ(h)=e^{Fh} and Qd(h)=∫_0^h e^{Fs} L Qc L^T e^{F^T s} ds
# for the OU per-axis chain [v,p,S,a]^T directly from F and L,
# then reduce to the SAME primitives used in the C++ code:
#   x=h/τ, α=e^{-x}, em1=α-1, em1_2=e^{-2x}-1,
#   A0=τ(1-α)=-τ em1, A1=τ²(-em1 - x α), A2=τ³(-2 em1 + α x(x+2)),
#   B0=-(τ/2) em1_2, and Ck = h^{k+1}/(k+1) for k=0..4
#
# Prints human-readable LaTeX for every step.

import sympy as sp

# ---------------- Utilities for pretty LaTeX output ----------------
def _bmatrix(lx: str) -> str:
    lx = lx.replace(r"\left[\begin{matrix}", r"\begin{bmatrix}")
    lx = lx.replace(r"\end{matrix}\right]", r"\end{bmatrix}")
    lx = lx.replace(r"\left[ \begin{matrix}", r"\begin{bmatrix}")
    lx = lx.replace(r"\end{matrix} \right]", r"\end{bmatrix}")
    return lx

def disp(title, expr=None, raw=None):
    print("\n% " + title)
    if raw is not None:
        print(raw)
        return
    lx = sp.latex(expr)
    print(r"\[ \displaystyle " + _bmatrix(lx) + r" \]")

# ---------------- Symbols ----------------
h, tau, sigma2 = sp.symbols('h tau sigma2', positive=True, real=True)
s, u = sp.symbols('s u', real=True, nonnegative=True)

# Per-axis OU chain: state x=[v,p,S,a]^T
# F is block upper-triangular [[A, B], [0, D]]
A = sp.Matrix([[0,0,0],
               [1,0,0],
               [0,1,0]])     # nilpotent (A^3=0)
B = sp.Matrix([[1],[0],[0]]) # a enters dv
D = sp.Matrix([[-1/tau]])    # da/dt = -(1/τ) a + w

# L selects only the a-channel for noise w (Qc scalar later)
L = sp.Matrix([[0],[0],[0],[1]])

# ---------------- Step 1: e^{F s} by block-triangular formula ----------------
# For F=[[A,B],[0,D]]:
#   e^{Fs} = [[ e^{A s}, ∫_0^s e^{A(s-u)} B e^{D u} du ],
#             [ 0,       e^{D s} ]]
I3 = sp.eye(3)
A2 = A*A

EA_s = I3 + A*s + A2*(s**2/2)        # e^{A s} because A^3=0
ED_s = sp.exp(-s/tau)                # e^{D s} (scalar)

# Top-right block E_TR(s) = ∫_0^s e^{A(s-u)} B e^{Du} du
TR_integrand = (I3 + A*(s-u) + A2*((s-u)**2/2)) * B * sp.exp(-u/tau)
E_TR = sp.Matrix(3,1, lambda i,j: sp.integrate(TR_integrand[i,0], (u,0,s)))

# Assemble E(s)
E_s = sp.Matrix.zeros(4,4)
E_s[0:3,0:3] = EA_s
E_s[0:3,3:4] = E_TR
E_s[3,3]     = ED_s

disp("Block formula for e^{Fs}", raw=r"""
Using the block upper-triangular formula for \(F=\begin{bmatrix}A&B\\0&D\end{bmatrix}\),
\[
e^{F s} \;=\;
\begin{bmatrix}
e^{A s} & \displaystyle\int_0^s e^{A(s-u)} B\,e^{D u}\,du\\[0.6em]
0       & e^{D s}
\end{bmatrix},
\qquad
e^{A s} = I + A s + \tfrac12 A^2 s^2,\quad e^{D s} = e^{-s/\tau}.
\]""")
disp("Computed e^{Fs}", E_s)

# ---------------- Step 2: Φ(h) = e^{F h} ----------------
Phi = sp.Matrix(E_s.subs(s, h))   # exact per definition
disp("Φ(h) = e^{F h}", Phi)

# Extract the four scalar couplings and present them with primitives
phi_va = sp.simplify(Phi[0,3])
phi_pa = sp.simplify(Phi[1,3])
phi_Sa = sp.simplify(Phi[2,3])
phi_aa = sp.simplify(Phi[3,3])

disp("Φ-axis block entries (coupling of a into v,p,S and a's own decay)", raw=r"""
Let \(x=\frac{h}{\tau}\), \(\alpha=e^{-x}\), \(\mathrm{em1}=\alpha-1\). Then
\[
\phi_{va}=\Phi_{1,4}=-\tau\,\mathrm{em1},\quad
\phi_{pa}=\Phi_{2,4}=\tau^2\,(x+\mathrm{em1}),\quad
\phi_{Sa}=\Phi_{3,4}=\tau^3\Big(\tfrac12 x^2 - x - \mathrm{em1}\Big),\quad
\phi_{aa}=\Phi_{4,4}=\alpha.
\]
(These are the closed forms after integrating the top-right block.)
""")

# ---------------- Step 3: Q_d(h) = ∫_0^h e^{Fs} L Qc L^T e^{F^T s} ds ----------------
# Continuous spectral density on a-channel: Qc = 2 σ^2 / τ (scalar)
qc = (2*sigma2)/tau

# C(s) = e^{F s} L = [C_top; C_bot]
C_top = E_TR            # 3×1
C_bot = sp.Matrix([[ED_s]])  # 1×1

# Build integrand = C(s) Qc C(s)^T
integrand = sp.zeros(4,4)
integrand[0:3,0:3] = qc * (C_top * C_top.T)
integrand[0:3,3:4] = qc * (C_top * C_bot.T)
integrand[3:4,0:3] = qc * (C_bot * C_top.T)
integrand[3,3]     = qc * (C_bot * C_bot.T)[0,0]

Qd = sp.Matrix(4,4, lambda i,j: sp.simplify(sp.integrate(integrand[i,j], (s,0,h))))
Qd = sp.simplify((Qd + Qd.T)/2)  # symmetry hygiene
disp("Q_d(h) = ∫_0^h e^{Fs} L Q_c L^T e^{F^T s} ds", Qd)

# ---------------- Step 4: Reduce Q_d to (2σ²/τ) * sym(K) with primitive A's and C's ----------------
# Define primitives (exactly as in C++)
x = sp.Symbol('x', positive=True, real=True)
alpha = sp.exp(-x)
em1   = alpha - 1
em1_2 = sp.exp(-2*x) - 1

A0 = -tau*em1                                   # τ(1-α)
A1 = tau**2 * (-em1 - x*alpha)                  # τ²(-em1 - x α)
A2 = tau**3 * (-2*em1 + alpha*x*(x+2))          # τ³(-2 em1 + α x(x+2))
B0 = -(tau/2)*em1_2                             # -τ/2 (e^{-2x}-1)

C0 = h
C1 = h**2/2
C2 = h**3/3
C3 = h**4/4
C4 = h**5/5

# Unit kernel K = (τ/(2σ²)) Qd, then symmetrize
K_raw = sp.simplify( (tau/(2*sigma2)) * Qd )
K_sym = sp.simplify( (K_raw + K_raw.T)/2 )

# Extract entries derived from F,L
Kvva = sp.simplify(K_sym[0,0])                # K_vv
Kvaa = sp.simplify(K_sym[0,3])                # K_va
Kppa = sp.simplify(K_sym[1,1])                # K_pp
Kpaa = sp.simplify(K_sym[1,3])                # K_pa
Kpva = sp.simplify(K_sym[0,1])                # K_pv
KSSa = sp.simplify(K_sym[2,2])                # K_SS
KSaa = sp.simplify(K_sym[2,3])                # K_Sa
KSpa = sp.simplify(K_sym[1,2])                # K_Sp
KSva = sp.simplify(K_sym[0,2])                # K_Sv
Kaaa = sp.simplify(K_sym[3,3])                # K_aa

# Build compact target forms (the EXACT ones your C++ uses), now as expressions of primitives
K_vv_t = tau**2 * ( C0 - 2*A0 + B0 )
K_va_t = tau * ( A0 - B0 )
K_pp_t = ( tau**2*C2 - 2*tau**3*C1 + 2*tau**3*A1 + tau**4*C0 - 2*tau**4*A0 + tau**4*B0 )
K_pa_t = ( tau*A1 + tau**2*(B0 - A0) )
K_pv_t = ( tau**2*(C1 - A1) - tau**3*(C0 - A0) + tau**3*(A0 - B0) )
K_SS_t = ( sp.Rational(1,4)*tau**2*C4 - tau**3*C3 + 2*tau**4*C2 - 2*tau**5*C1 + tau**6*C0
         - tau**4*A2 + 2*tau**5*A1 - 2*tau**6*A0 + tau**6*B0 )
K_Sa_t = ( sp.Rational(1,2)*tau*A2 - tau**2*A1 + tau**3*A0 - tau**3*B0 )
K_Sp_t = ( sp.Rational(1,2)*tau**2*C3 - sp.Rational(3,2)*tau**3*C2 + 2*tau**4*C1 - tau**5*C0
         + sp.Rational(1,2)*tau**3*A2 - 2*tau**4*A1 + 2*tau**5*A0 - tau**5*B0 )
K_Sv_t = ( sp.Rational(1,2)*tau**2*(C2 - A2) - tau**3*(C1 - A1) + tau**4*(C0 - A0)
         - tau**4*(A0 - B0) )
K_aa_t = B0

K_target = sp.Matrix([
    [K_vv_t, K_pv_t, K_Sv_t, K_va_t],
    [K_pv_t, K_pp_t, K_Sp_t, K_pa_t],
    [K_Sv_t, K_Sp_t, K_SS_t, K_Sa_t],
    [K_va_t, K_pa_t, K_Sa_t, K_aa_t]
])

# ---------------- Step 5: Print step-by-step LaTeX explanations ----------------
disp("Define non-dimensional primitives", raw=r"""
We normalize by \(x=\tfrac{h}{\tau}\) and define
\[
\alpha = e^{-x},\quad \mathrm{em1}=\alpha-1,\quad \mathrm{em1}_2=e^{-2x}-1.
\]
We also use the short-hands (matching code):
\[
A_0=\tau(1-\alpha)=-\tau\,\mathrm{em1},\quad
A_1=\tau^2(-\mathrm{em1}-x\alpha),\quad
A_2=\tau^3(-2\,\mathrm{em1}+\alpha x(x+2)),\quad
B_0=-\tfrac{\tau}{2}\,\mathrm{em1}_2,
\]
\[
C_0=h,\quad C_1=\tfrac{h^2}{2},\quad C_2=\tfrac{h^3}{3},\quad C_3=\tfrac{h^4}{4},\quad C_4=\tfrac{h^5}{5}.
\]
""")

disp("K(h,τ) derived from F and L (symmetrized unit kernel: K = (τ/2σ²) Q_d)", K_sym)
disp("Target compact form for K(h,τ) in primitives A_•, B_0, C_•", K_target)

def eq_latex(name, lhs, rhs):
    ok = sp.simplify(lhs - rhs) == 0
    tick = r"\quad(\checkmark)" if ok else r"\quad(\text{mismatch!})"
    print(r"\[ %s:\quad %s \;=\; %s %s \]" % (name, sp.latex(lhs), sp.latex(rhs), tick))

print("\n% === Entry-by-entry verification (symbolic) ===")
eq_latex("K_{vv}", Kvva, K_vv_t)
eq_latex("K_{va}", Kvaa, K_va_t)
eq_latex("K_{pp}", Kppa, K_pp_t)
eq_latex("K_{pa}", Kpaa, K_pa_t)
eq_latex("K_{pv}", Kpva, K_pv_t)
eq_latex("K_{SS}", KSSa, K_SS_t)
eq_latex("K_{Sa}", KSaa, K_Sa_t)
eq_latex("K_{Sp}", KSpa, K_Sp_t)
eq_latex("K_{Sv}", KSva, K_Sv_t)
eq_latex("K_{aa}", Kaaa, K_aa_t)

disp("Final discrete covariance", raw=r"""
From the unit kernel we recover the discrete covariance:
\[
Q_{d,\text{axis}}(h)\;=\;\frac{2\sigma^2}{\tau}\;\mathrm{sym}\,K(h,\tau),
\qquad \mathrm{sym}\,K=\tfrac12\bigl(K+K^\top\bigr).
\]
This is exactly the form used in the C++ \texttt{QdAxis4x1\_analytic} general branch.
""")

disp("Per-axis transition Φ_{axis}(h) in compact form", raw=r"""
With \(x=\frac{h}{\tau}\), \(\alpha=e^{-x}\), \(\mathrm{em1}=\alpha-1\),
\[
\Phi_{\text{axis}}(h)=
\begin{bmatrix}
1 & 0 & 0 & -\tau\,\mathrm{em1}\\[2pt]
h & 1 & 0 & \tau^2(x+\mathrm{em1})\\[2pt]
\frac{h^2}{2} & h & 1 & \tau^3\bigl(\tfrac12 x^2 - x - \mathrm{em1}\bigr)\\[2pt]
0 & 0 & 0 & \alpha
\end{bmatrix}.
\]
This matches the code’s \texttt{PhiAxis4x1\_analytic}.
""")
