# -*- coding: utf-8 -*-
# SymPy derivation: Φ(h)=e^{Fh} and Qd(h)=∫ e^{Fs} L Qc L^T e^{F^T s} ds
# for the per-axis chain x=[v,p,S,a]^T with OU acceleration and nilpotent kinematics.
# Prints a self-contained, human-readable LaTeX section ready for your paper.

import sympy as sp
import mpmath as mp
import random

# ---------------- Utilities ----------------
def _bmatrix(lx: str) -> str:
    lx = lx.replace(r"\left[\begin{matrix}", r"\begin{bmatrix}")
    lx = lx.replace(r"\end{matrix}\right]", r"\end{bmatrix}")
    lx = lx.replace(r"\left[ \begin{matrix}", r"\begin{bmatrix}")
    lx = lx.replace(r"\end{matrix} \right]", r"\end{bmatrix}")
    return lx

def latex_b(expr) -> str:
    return _bmatrix(sp.latex(expr))

def is_zero_expr(expr):
    z = sp.simplify(sp.together(sp.factor(expr)))
    if z == 0 or getattr(z, "is_zero", False):
        return True
    vars_ = sorted(list(z.free_symbols), key=lambda s: s.name)
    if not vars_:
        try:
            val = complex(sp.N(z))
            return abs(val) <= 1e-10
        except Exception:
            return False
    f = sp.lambdify(vars_, z, modules="mpmath")
    for _ in range(4):
        subs = []
        for v in vars_:
            name = v.name
            if name == "h":
                subs.append(mp.mpf("0.2") + mp.mpf(random.random()))
            elif name == "tau":
                subs.append(mp.mpf("0.5") + mp.mpf(random.random())*mp.mpf("2.5"))
            elif name == "sigma2":
                subs.append(mp.mpf("0.3") + mp.mpf(random.random())*mp.mpf("1.7"))
            else:
                subs.append(mp.mpf(random.random()))
        try:
            val = f(*subs)
            if abs(complex(val)) > 1e-9:
                return False
        except Exception:
            return False
    return True

def eq_line(name, lhs, rhs):
    ok = is_zero_expr(sp.simplify(lhs - rhs))
    tick = r"\checkmark" if ok else r"\text{mismatch!}"
    return r"\[ {}:\; {} \;=\; {} \quad ({}) \]".format(
        name, sp.latex(sp.simplify(lhs)), sp.latex(sp.simplify(rhs)), tick
    )

# ---------------- Symbols ----------------
h, tau, sigma2 = sp.symbols('h tau sigma2', positive=True, real=True)
s, u = sp.symbols('s u', real=True, nonnegative=True)

# Per-axis OU chain: state x=[v,p,S,a]^T
A = sp.Matrix([[0,0,0],
               [1,0,0],
               [0,1,0]])     # A^3=0
B = sp.Matrix([[1],[0],[0]])
D = sp.Matrix([[-1/tau]])
L = sp.Matrix([[0],[0],[0],[1]])

I3 = sp.eye(3)
A2 = A*A

# e^{F s} by block upper-triangular formula
EA_s = I3 + A*s + A2*(s**2/2)
ED_s = sp.exp(-s/tau)

TR_integrand = (I3 + A*(s-u) + A2*((s-u)**2/2)) * B * sp.exp(-u/tau)
E_TR = sp.Matrix(3,1, lambda i,j: sp.integrate(TR_integrand[i,0], (u,0,s)))

E_s = sp.Matrix.zeros(4,4)
E_s[0:3,0:3] = EA_s
E_s[0:3,3:4] = E_TR
E_s[3,3]     = ED_s

Phi = sp.Matrix(E_s.subs(s, h))

# Q_d
qc = (2*sigma2)/tau
C_top = E_TR
C_bot = sp.Matrix([[ED_s]])

integrand = sp.zeros(4,4)
integrand[0:3,0:3] = qc * (C_top * C_top.T)
integrand[0:3,3:4] = qc * (C_top * C_bot.T)
integrand[3:4,0:3] = qc * (C_bot * C_top.T)
integrand[3,3]     = qc * (C_bot * C_bot.T)[0,0]

Qd = sp.Matrix(4,4, lambda i,j: sp.simplify(sp.integrate(integrand[i,j], (s,0,h))))
Qd = sp.simplify((Qd + Qd.T)/2)

# Primitives and K target
x = sp.Symbol('x', positive=True, real=True)
alpha = sp.exp(-x)
em1   = alpha - 1
em1_2 = sp.exp(-2*x) - 1

A0 = -tau*em1
A1 = tau**2 * (-em1 - x*alpha)
A2 = tau**3 * (-2*em1 + alpha*x*(x+2))
B0 = -(tau/2)*em1_2

C0 = h
C1 = h**2/2
C2 = h**3/3
C3 = h**4/4
C4 = h**5/5

K_raw = sp.simplify( (tau/(2*sigma2)) * Qd )
K_sym = sp.simplify( (K_raw + K_raw.T)/2 )

K_vv_t = tau**2 * ( C0 - 2*A0 + B0 )
K_va_t = tau * ( A0 - B0 )
K_pp_t = ( tau**2*C2 - 2*tau**3*C1 + 2*tau**3*A1 + tau**4*C0 - 2*tau**4*A0 + tau**4*B0 )
K_pa_t = ( tau*A1 + tau**2*(B0 - A0) )
K_SS_t = ( sp.Rational(1,4)*tau**2*C4 - tau**3*C3 + 2*tau**4*C2 - 2*tau**5*C1 + tau**6*C0
         - tau**4*A2 + 2*tau**5*A1 - 2*tau**6*A0 + tau**6*B0 )
K_Sp_t = ( sp.Rational(1,2)*tau**2*C3 - sp.Rational(3,2)*tau**3*C2 + 2*tau**4*C1 - tau**5*C0
         + sp.Rational(1,2)*tau**3*A2 - 2*tau**4*A1 + 2*tau**5*A0 - tau**5*B0 )
K_Sv_t = ( sp.Rational(1,2)*tau**2*(C2 - A2) - tau**3*(C1 - A1) + tau**4*(C0 - A0)
         - tau**4*(A0 - B0) )
K_aa_t = B0

K_target = sp.Matrix([
    [K_vv_t, K_Sv_t*0 + (tau**2*(C1 - A1) - tau**3*(C0 - A0) + tau**3*(A0 - B0)), K_Sv_t, K_va_t],
    [K_Sv_t*0 + (tau**2*(C1 - A1) - tau**3*(C0 - A0) + tau**3*(A0 - B0)), K_pp_t, K_Sp_t, K_pa_t],
    [K_Sv_t, K_Sp_t, K_SS_t, K_Sa_t := ( sp.Rational(1,2)*tau*A2 - tau**2*A1 + tau**3*A0 - tau**3*B0 )],
    [K_va_t, K_pa_t, K_Sa_t, K_aa_t]
])

K_target_ht = sp.simplify(K_target.subs({x: h/tau}))
K_target_sym = sp.simplify((K_target_ht + K_target_ht.T)/2)

# ---------------- LaTeX output ----------------

print(r"""
% ===================== BEGIN: Step-by-step discretization (paste into paper) =====================

\section*{Analytic Discretization from $(F,L)$ to $(\Phi,Q_d)$: A Step-by-Step Guide}

\paragraph{1. Block structure (one axis).}
For each axis we stack the linear states
\[
\boldsymbol{x}_{\text{axis}}=\begin{bmatrix} v & p & S & a \end{bmatrix}^{\!\top},
\]
with continuous-time dynamics
\[
\frac{d}{dt}\!\begin{bmatrix} v\\ p\\ S\\ a\end{bmatrix}
=
\underbrace{\begin{bmatrix}
A & B\\[2pt]
0 & D
\end{bmatrix}}_{F}
\begin{bmatrix} v\\ p\\ S\\ a\end{bmatrix}
+
\underbrace{\begin{bmatrix} 0\\0\\0\\1\end{bmatrix}}_{L}\,w(t),
\qquad
\begin{aligned}
&A=\begin{bmatrix}0&0&0\\ 1&0&0\\ 0&1&0\end{bmatrix},\;\;
B=\begin{bmatrix}1\\0\\0\end{bmatrix},\\
&D=-\tfrac{1}{\tau},\quad
\mathbb E[w(t)w(s)]=q_c\,\delta(t-s),\;\; q_c=\tfrac{2\sigma^2}{\tau}.
\end{aligned}
\]
Here $A^3=0$ (nilpotent kinematics) and $a$ is OU.

\paragraph{2. Transition matrix via block-triangular exponential.}
Since $F$ is block upper triangular,
\[
e^{F s}
=
\begin{bmatrix}
e^{A s} & \displaystyle\int_{0}^{s} e^{A(s-u)}\,B\,e^{-u/\tau}\,du\\[6pt]
0 & e^{-s/\tau}
\end{bmatrix},
\qquad
e^{A s}=I + A s + \tfrac{1}{2}A^2 s^2 .
\]
Setting $s=h$, the per-axis transition is
\[
\Phi_{\text{axis}}(h)=
\begin{bmatrix}
1 & 0 & 0 & \phi_{va}\\
h & 1 & 0 & \phi_{pa}\\
\frac{h^2}{2} & h & 1 & \phi_{Sa}\\
0 & 0 & 0 & \phi_{aa}
\end{bmatrix}.
\]

\paragraph{3. Closed-form feedthroughs in compact primitives.}
Let $x=\tfrac{h}{\tau}$, $\alpha=e^{-x}$, $\mathrm{em1}=\alpha-1$.
Then
\[
\boxed{
\phi_{va}=-\tau\,\mathrm{em1},\qquad
\phi_{pa}=\tau^2\,(x+\mathrm{em1}),\qquad
\phi_{Sa}=\tau^3\!\Big(\tfrac{1}{2}x^2 - x - \mathrm{em1}\Big),\qquad
\phi_{aa}=\alpha.}
\]

\paragraph{4. Discrete covariance $Q_d$ as noise-flow integral.}
With $q_c=\tfrac{2\sigma^2}{\tau}$,
\[
Q_d(h)=\int_{0}^{h} e^{F s} L q_c L^{\!\top} e^{F^{\!\top} s}\,ds
=\int_{0}^{h} q_c\,
\begin{bmatrix}
C_{\mathrm{top}}(s)\\ e^{-s/\tau}
\end{bmatrix}
\begin{bmatrix}
C_{\mathrm{top}}(s)^{\!\top} & e^{-s/\tau}
\end{bmatrix}\,ds,
\]
where $C_{\mathrm{top}}(s)=\int_{0}^{s} e^{A(s-u)}B\,e^{-u/\tau}du$.
Introduce kinematic monomials $C_0=h,\,C_1=\tfrac{h^2}{2},\,C_2=\tfrac{h^3}{3},\,C_3=\tfrac{h^4}{4},\,C_4=\tfrac{h^5}{5}$ and OU primitives
\[
\mathrm{em1}_2=e^{-2x}-1,\quad
A_0=-\tau\,\mathrm{em1},\quad
A_1=\tau^2(-\mathrm{em1}-x\alpha),\quad
A_2=\tau^3\!\big(-2\,\mathrm{em1}+\alpha x(x+2)\big),\quad
B_0=-\tfrac{\tau}{2}\,\mathrm{em1}_2 .
\]
Then, factoring $q_c$, we define the unit kernel $K(h,\tau)$ by
\[
Q_{d,\text{axis}}(h)=\frac{2\sigma^2}{\tau}\;\mathrm{sym}\,K(h,\tau),
\qquad \mathrm{sym}\,K=\tfrac12\big(K+K^\top\big).
\]

\paragraph{5. Kernel $K(h,\tau)$ entries (compact, code-ready).}
\[
K(h,\tau)=
\begin{bmatrix}
K_{vv} & K_{pv} & K_{Sv} & K_{va}\\
K_{pv} & K_{pp} & K_{Sp} & K_{pa}\\
K_{Sv} & K_{Sp} & K_{SS} & K_{Sa}\\
K_{va} & K_{pa} & K_{Sa} & K_{aa}
\end{bmatrix},
\]
with
\[
\begin{aligned}
K_{vv}&=\tau^2\Big(C_0-2A_0+B_0\Big),\qquad
&K_{va}&=\tau\Big(A_0-B_0\Big),\\
K_{pp}&=\tau^2 C_2 - 2\tau^3 C_1 + 2\tau^3 A_1 + \tau^4 C_0 - 2\tau^4 A_0 + \tau^4 B_0,\quad
&K_{pa}&=\tau A_1 + \tau^2(B_0 - A_0),\\
K_{SS}&=\tfrac{1}{4}\tau^2 C_4 - \tau^3 C_3 + 2\tau^4 C_2 - 2\tau^5 C_1 + \tau^6 C_0
        - \tau^4 A_2 + 2\tau^5 A_1 - 2\tau^6 A_0 + \tau^6 B_0,\\
K_{Sp}&=\tfrac{1}{2}\tau^2 C_3 - \tfrac{3}{2}\tau^3 C_2 + 2\tau^4 C_1 - \tau^5 C_0
        + \tfrac{1}{2}\tau^3 A_2 - 2\tau^4 A_1 + 2\tau^5 A_0 - \tau^5 B_0,\\
K_{Sv}&=\tfrac{1}{2}\tau^2(C_2 - A_2) - \tau^3(C_1 - A_1) + \tau^4(C_0 - A_0)
        - \tau^4(A_0 - B_0),\\
K_{aa}&=B_0.
\end{aligned}
\]

\paragraph{6. Small-step asymptotics ($x\ll 1$) for numerical stability.}
\[
\alpha=1-x+\tfrac{x^2}{2}-\tfrac{x^3}{6}+\cdots,\quad
\mathrm{em1}=-x+\tfrac{x^2}{2}-\tfrac{x^3}{6}+\cdots,\quad
\mathrm{em1}_2=-2x+2x^2-\tfrac{4}{3}x^3+\cdots.
\]

\paragraph{7. Kronecker assembly (3-axis, shared $\tau$).}
Let $\Sigma_{aw}\in\mathbb{R}^{3\times3}$ be the stationary covariance (full SPD).
Then for $[v(3),p(3),S(3),a(3)]$,
\[
\Phi_{LL}(h)=I_3\otimes \Phi_{\text{axis}}(h),
\qquad
Q_{LL}(h)=\frac{2}{\tau}\;\Sigma_{aw}\otimes \mathrm{sym}\,K(h,\tau).
\]

\paragraph{8. Plugging into the Kalman predictor.}
\[
\hat{\boldsymbol{x}}_{k|k-1}=\Phi\,\hat{\boldsymbol{x}}_{k-1|k-1},\qquad
P_{k|k-1}=\Phi\,P_{k-1|k-1}\,\Phi^{\!\top}+Q_d.
\]
The attitude block is advanced by the standard Q-MEKF constant-$\boldsymbol{\omega}$ step,
the linear block by $\Phi_{LL},Q_{LL}$, and updates use Joseph form with PSD projection.

% ===================== END: Step-by-step discretization =====================
""")

# ---------------- Verification lines (short, after LaTeX) ----------------
print("% === Verification: K from integral equals compact primitives (entry-wise) ===")
print(eq_line("K_{vv}", K_sym[0,0], K_target_sym[0,0]))
print(eq_line("K_{pv}", K_sym[0,1], K_target_sym[0,1]))
print(eq_line("K_{Sv}", K_sym[0,2], K_target_sym[0,2]))
print(eq_line("K_{va}", K_sym[0,3], K_target_sym[0,3]))
print(eq_line("K_{pp}", K_sym[1,1], K_target_sym[1,1]))
print(eq_line("K_{Sp}", K_sym[1,2], K_target_sym[1,2]))
print(eq_line("K_{pa}", K_sym[1,3], K_target_sym[1,3]))
print(eq_line("K_{SS}", K_sym[2,2], K_target_sym[2,2]))
print(eq_line("K_{Sa}", K_sym[2,3], K_target_sym[2,3]))
print(eq_line("K_{aa}", K_sym[3,3], K_target_sym[3,3]))
