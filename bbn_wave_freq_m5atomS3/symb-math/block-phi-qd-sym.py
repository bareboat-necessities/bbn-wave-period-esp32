# -*- coding: utf-8 -*-
# Step-by-step symbolic derivation of Φ(h)=exp(F h) and Qd(h)=∫_0^h e^{F s} L Qc L^T e^{F^T s} ds
# for the OU-driven per-axis chain x=[v,p,S,a]^T with
#   dv = a ,  dp = v ,  dS = p ,  da = -(1/τ) a + w ,  E[w w^T] = Qc dt,  Qc = 2 σ^2 / τ (scalar).
#
# The script:
#   1) Builds F and L.
#   2) Derives e^{As} using nilpotency of A (A^3=0) and uses block-triangular exponential:
#        e^{[[A,B],[0,D]] s} = [[e^{A s}, ∫_0^s e^{A(s-u)} B e^{D u} du],[0, e^{D s}]]
#   3) Forms E(s)L Qc L^T E(s)^T, integrates over s to get Qd(h).
#   4) Rewrites results in the same primitives and scalar shorthands used in the C++ code:
#        x=h/τ, α=e^{-x}, em1=α-1, em1_2=e^{-2x}-1,
#        φ_va, φ_pa, φ_Sa  and K entries (vv,pv,Sv,va,pp,Sp,pa,SS,Sa,aa).
#   5) Prints LaTeX at each conceptual step (ready to copy-paste).

import sympy as sp

# ----------------- basic symbols -----------------
h, s, u, tau, sigma2 = sp.symbols('h s u tau sigma2', positive=True, real=True)

# ----------------- build F and L -----------------
# State order per axis: x = [v, p, S, a]^T
A = sp.Matrix([[0,0,0],
               [1,0,0],
               [0,1,0]])               # nilpotent (A^3=0)
B = sp.Matrix([[1],[0],[0]])           # a enters dv
D = sp.Matrix([[-1/tau]])              # OU drift on a

F = sp.Matrix([[0,0,0,1],              # [ [A, B],
               [1,0,0,0],              #   [0, D] ]
               [0,1,0,0],
               [0,0,0,-1/tau]])

# Process-noise input only on 'a' channel:
L = sp.Matrix([[0],[0],[0],[1]])       # w enters 'a' only
Qc = sp.Matrix([[2*sigma2/tau]])       # scalar (2 σ^2 / τ)

# ----------------- helpers for LaTeX pretty matrices -----------------
def bmatrix(expr):
    return sp.latex(expr).replace(r"\left[\begin{matrix}", r"\begin{bmatrix}") \
                         .replace(r"\end{matrix}\right]", r"\end{bmatrix}")

def display_latex(title, expr):
    print("\n% " + title)
    print(r"\[ \displaystyle " + bmatrix(expr) + r" \]")

def display_latex_text(block):
    print("\n" + block)

# ----------------- Step 1: print F and L -----------------
display_latex_text(r"% ===== Step 1: Continuous-time LTI SDE blocks =====")
display_latex("F = [[A,B],[0,D]]", F)
display_latex("L (a-channel)", L)
display_latex("Q_c (scalar)", Qc)

# ----------------- Step 2: e^{F s} by block-triangular exponential -----------------
# Because A is nilpotent: e^{A s} = I + A s + 1/2 A^2 s^2 .
I3 = sp.eye(3)
A2 = A*A
expAs = I3 + A*s + A2*(s**2/2)              # exact (since A^3=0)
expDs = sp.exp(-s/tau)

# Upper-right block: ∫_0^s e^{A (s-u)} B e^{D u} du
TR_integrand = (I3 + A*(s-u) + A2*((s-u)**2/2)) * B * sp.exp(-u/tau)
expF_TR = sp.Matrix(3,1, lambda i,j: sp.integrate(TR_integrand[i,0], (u,0,s)))

# Assemble e^{F s} = E(s)
E = sp.Matrix.zeros(4,4)
E[0:3,0:3] = expAs
E[0:3,3:4] = expF_TR
E[3,3]     = expDs

display_latex_text(r"% ===== Step 2: Block exponential for the triangular F =====")
display_latex(r"e^{A s} = I + A s + \tfrac{1}{2}A^2 s^2", expAs)
display_latex(r"\int_0^s e^{A (s-u)} B e^{D u}\,du", expF_TR)
display_latex(r"E(s)=e^{F s} = \begin{bmatrix} e^{A s} & \int_0^s e^{A(s-u)} B e^{D u} du \\ 0 & e^{D s} \end{bmatrix}", E)

# ----------------- Step 3: Φ(h) = E(h) -----------------
Phi_axis = E.subs(s, h)
display_latex_text(r"% ===== Step 3: Discrete transition Φ(h) =====")
display_latex(r"\Phi(h)=e^{F h}", Phi_axis)

# ----------------- Step 4: Q_d(h) via covariance integral -----------------
# Q_d(h) = ∫_0^h E(s) L Qc L^T E(s)^T ds .
C = E * L                       # 4×1
integrand = C * Qc[0,0] * C.T   # 4×4
Qd = sp.Matrix(4,4, lambda i,j: sp.integrate(integrand[i,j], (s,0,h)))
Qd = sp.simplify((Qd + Qd.T)/2)

display_latex_text(r"% ===== Step 4: Discrete-time covariance via integral =====")
display_latex(r"C(s) = E(s)L", C)
display_latex(r"\int_0^h C(s) Q_c C(s)^{\!\top} ds", Qd)

# ----------------- Step 5: Rewrite to C++ primitives and scalars -----------------
# Define primitives and rewrite
x = sp.symbols('x', positive=True, real=True)       # x = h/tau
alpha = sp.exp(-x)                                  # e^{-x}
em1   = alpha - 1                                   # exp(-x) - 1
em1_2 = sp.exp(-2*x) - 1                            # exp(-2x) - 1

subs_small = {
    sp.exp(-h/tau): alpha,
    sp.exp(-2*h/tau): sp.exp(-2*x),
    h/tau: x
}
# replace higher exponentials
Qd_rew = sp.simplify(Qd.xreplace(subs_small))
Phi_rew = sp.simplify(Phi_axis.xreplace(subs_small))

# Now express the key scalars exactly as in the C++:
A0 = -tau*em1
A1 = tau**2 * (-em1 - x*alpha)
A2 = tau**3 * (-2*em1 + alpha*x*(x+2))
B0 = -(tau/2)*em1_2

phi_va = -tau*em1
phi_pa = tau**2 * (x + em1)
phi_Sa = tau**3 * (sp.Rational(1,2)*x**2 - x - em1)
phi_aa = alpha

Phi_cpp = sp.Matrix([
    [1,        0,     0,     phi_va],
    [h,        1,     0,     phi_pa],
    [h**2/2,   h,     1,     phi_Sa],
    [0,        0,     0,     phi_aa]
])

# Build K entries from the *derived* Qd by factoring out (2 σ^2 / τ)
K_from_Qd = sp.simplify((tau/(2*sigma2)) * Qd_rew)

# Extract labeled entries to show the match to code:
K_vv = sp.simplify(K_from_Qd[0,0])
K_pv = sp.simplify(K_from_Qd[1,0])
K_Sv = sp.simplify(K_from_Qd[2,0])
K_va = sp.simplify(K_from_Qd[0,3])

K_pp = sp.simplify(K_from_Qd[1,1])
K_Sp = sp.simplify(K_from_Qd[2,1])
K_pa = sp.simplify(K_from_Qd[1,3])

K_SS = sp.simplify(K_from_Qd[2,2])
K_Sa = sp.simplify(K_from_Qd[2,3])

K_aa = sp.simplify(K_from_Qd[3,3])

display_latex_text(r"% ===== Step 5: Rewrite into C++ primitives (x, alpha, em1, em1_2) and C++ scalars =====")
display_latex_text(r"""
% Primitives:
\[
x=\frac{h}{\tau},\qquad \alpha=e^{-x},\qquad \mathrm{em1}=\alpha-1,\qquad \mathrm{em1}_2=e^{-2x}-1.
\]
% Transition scalars (exactly as in C++):
\[
\phi_{va}=-\tau\,\mathrm{em1},\qquad
\phi_{pa}=\tau^2\,(x+\mathrm{em1}),\qquad
\phi_{Sa}=\tau^3\!\left(\tfrac{1}{2}x^2-x-\mathrm{em1}\right),\qquad
\phi_{aa}=\alpha.
\]
""")
display_latex(r"\Phi_{\text{axis}}(h)\ \text{(rewritten)}", Phi_cpp)

display_latex_text(r"""
% Discrete covariance structure:
% Factor out Q_d(h) = (2\sigma^2/\tau) * K(h,\tau), then symmetrize in code.
% Below are the K entries as derived from the integral:
""")

def emit_named(name, expr):
    print(r"\[%s \;=\; %s\]" % (name, sp.latex(sp.simplify(expr)
          ).replace(r"\left[\begin{matrix}", r"\begin{bmatrix}"
          ).replace(r"\end{matrix}\right]", r"\end{bmatrix}")))

emit_named("K_{vv}", K_vv)
emit_named("K_{pv}", K_pv)
emit_named("K_{Sv}", K_Sv)
emit_named("K_{va}", K_va)
emit_named("K_{pp}", K_pp)
emit_named("K_{Sp}", K_Sp)
emit_named("K_{pa}", K_pa)
emit_named("K_{SS}", K_SS)
emit_named("K_{Sa}", K_Sa)
emit_named("K_{aa}", K_aa)

display_latex_text(r"""
% Final assembly (exactly the C++ pattern):
\[
Q_{d,\text{axis}}(h) \;=\; \frac{2\,\sigma^2}{\tau}\,\frac{K(h,\tau)+K(h,\tau)^\top}{2},
\quad\text{with}\;
K(h,\tau)=
\begin{bmatrix}
K_{vv} & K_{pv} & K_{Sv} & K_{va}\\
K_{pv} & K_{pp} & K_{Sp} & K_{pa}\\
K_{Sv} & K_{Sp} & K_{SS} & K_{Sa}\\
K_{va} & K_{pa} & K_{Sa} & K_{aa}
\end{bmatrix}.
\]
""")
