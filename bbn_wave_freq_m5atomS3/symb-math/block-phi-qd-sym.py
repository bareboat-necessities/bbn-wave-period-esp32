# -*- coding: utf-8 -*-
# SymPy derivation of Φ(h)=e^{Fh} and Qd(h)=∫_0^h e^{Fs} L Qc L^T e^{F^T s} ds
# for the per-axis OU-driven chain x=[v,p,S,a]^T, reduced to the SAME
# compact primitives used in the C++ code (alpha, em1, em1_2, A0..A2, B0, Ck).
# It also emits a human-readable LaTeX write-up (step-by-step explanation),
# prints Φ_axis(h), K(h,τ), Qd_axis(h) relations, verifies entries, and
# provides Kronecker assembly for the 3-axis case with shared τ.
#
# Output: LaTeX only (to stdout). Paste directly into your paper.

import sympy as sp

# ---------------- Utilities ----------------

def say(tex: str):
    """Print raw LaTeX text (paragraphs, section headers, etc.)."""
    print(tex)

def _bmatrix(lx: str) -> str:
    # convert \left[\begin{matrix} ... \end{matrix}\right] to bmatrix for readability
    lx = lx.replace(r"\left[\begin{matrix}", r"\begin{bmatrix}")
    lx = lx.replace(r"\end{matrix}\right]", r"\end{bmatrix}")
    lx = lx.replace(r"\left[ \begin{matrix}", r"\begin{bmatrix}")
    lx = lx.replace(r"\end{matrix} \right]", r"\end{bmatrix}")
    return lx

def disp_math(expr, inline=False):
    """Display a SymPy expression as LaTeX block math or inline."""
    lx = sp.latex(expr)
    lx = _bmatrix(lx)
    if inline:
        print(r"$%s$" % lx)
    else:
        print(r"\[ \displaystyle %s \]" % lx)

def disp_block(title, expr=None, raw=None):
    print("\n% " + title)
    if raw is not None:
        print(raw)
    elif expr is not None:
        disp_math(expr)

# Robust zero check with numeric fallback
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
    import random, mpmath as mp
    for _ in range(4):
        subs = []
        for v in vars_:
            name = v.name
            if name == "h":
                subs.append(mp.mpf("0.25") + mp.mpf(random.random()) * mp.mpf("0.75"))
            elif name == "tau":
                subs.append(mp.mpf("0.4") + mp.mpf(random.random()) * mp.mpf("3.0"))
            elif name == "sigma2":
                subs.append(mp.mpf("0.2") + mp.mpf(random.random()) * mp.mpf("2.0"))
            else:
                subs.append(mp.mpf(random.random()))
        try:
            val = f(*subs)
            if abs(complex(val)) > 1e-9:
                return False
        except Exception:
            return False
    return True

def eq_print(name, lhs, rhs):
    L = sp.simplify(lhs)
    R = sp.simplify(rhs)
    ok = is_zero_expr(L - R)
    tick = r"\;(\checkmark)" if ok else r"\;(\text{mismatch!})"
    print(r"\[%s:\;\; %s \;=\; %s %s\]" % (name, sp.latex(L), sp.latex(R), tick))

# ---------------- Symbols & per-axis model ----------------

h, tau, sigma2 = sp.symbols('h tau sigma2', positive=True, real=True)
s, u = sp.symbols('s u', real=True, nonnegative=True)

# Per-axis OU chain: x=[v,p,S,a]^T
A = sp.Matrix([[0,0,0],
               [1,0,0],
               [0,1,0]])     # nilpotent (A^3 = 0)
B = sp.Matrix([[1],[0],[0]]) # a -> dv
D = sp.Matrix([[-1/tau]])    # da/dt = -(1/τ) a + w
L = sp.Matrix([[0],[0],[0],[1]])  # noise enters a only

I3 = sp.eye(3)
A2 = A*A

# e^{Fs} via block upper triangular form
EA_s = I3 + A*s + A2*(s**2/2)       # exact since A^3=0
ED_s = sp.exp(-s/tau)

TR_integrand = (I3 + A*(s-u) + A2*((s-u)**2/2)) * B * sp.exp(-u/tau)
E_TR = sp.Matrix(3,1, lambda i,j: sp.integrate(TR_integrand[i,0], (u,0,s)))

E_s = sp.Matrix.zeros(4,4)
E_s[0:3,0:3] = EA_s
E_s[0:3,3:4] = E_TR
E_s[3,3]     = ED_s

Phi = sp.Matrix(E_s.subs(s, h))  # per-axis transition

# Qd(h): noise-flow integral
qc = (2*sigma2)/tau
C_top = E_TR
C_bot = sp.Matrix([[ED_s]])

integrand = sp.zeros(4,4)
integrand[0:3,0:3] = qc * (C_top * C_top.T)
integrand[0:3,3:4] = qc * (C_top * C_bot.T)
integrand[3:4,0:3] = qc * (C_bot * C_top.T)
integrand[3,3]     = qc * (C_bot * C_bot.T)[0,0]

Qd = sp.Matrix(4,4, lambda i,j: sp.simplify(sp.integrate(integrand[i,j], (s,0,h))))
Qd = sp.simplify((Qd + Qd.T)/2)  # Joseph symmetry hygiene

# ---------------- Compact primitives and target K(h,τ) ----------------

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

# Unit kernel K: Qd_axis(h) = (2σ^2/τ) * sym(K)
K_raw = sp.simplify( (tau/(2*sigma2)) * Qd )
K_sym = sp.simplify( (K_raw + K_raw.T)/2 )

# Target compact K in the same primitives (matches C++)
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
    [K_vv_t, K_Sv_t.subs({C2:C2, A2:A2, C1:C1, A1:A1, C0:C0, A0:A0, B0:B0})*0 + (tau**2*(C1 - A1) - tau**3*(C0 - A0) + tau**3*(A0 - B0)), K_Sv_t, K_va_t],
    [K_Sv_t.subs({C2:C2, A2:A2, C1:C1, A1:A1, C0:C0, A0:A0, B0:B0})*0 + (tau**2*(C1 - A1) - tau**3*(C0 - A0) + tau**3*(A0 - B0)), K_pp_t, K_Sp_t, K_pa_t],
    [K_Sv_t, K_Sp_t, K_SS_t, K_Sa_t:= ( sp.Rational(1,2)*tau*A2 - tau**2*A1 + tau**3*A0 - tau**3*B0 )],
    [K_va_t, K_pa_t, K_Sa_t, K_aa_t]
])

# To keep K_target exactly as in the paper/code, rebuild explicitly:
K_target = sp.Matrix([
    [K_vv_t,                       (tau**2*(C1 - A1) - tau**3*(C0 - A0) + tau**3*(A0 - B0)),  K_Sv_t, K_va_t],
    [(tau**2*(C1 - A1) - tau**3*(C0 - A0) + tau**3*(A0 - B0)), K_pp_t,                       K_Sp_t, K_pa_t],
    [K_Sv_t,                       K_Sp_t,                       K_SS_t,  ( sp.Rational(1,2)*tau*A2 - tau**2*A1 + tau**3*A0 - tau**3*B0 )],
    [K_va_t,                       K_pa_t,                       ( sp.Rational(1,2)*tau*A2 - tau**2*A1 + tau**3*A0 - tau**3*B0 ), K_aa_t]
])

K_target_ht = sp.simplify(K_target.subs({x: h/tau}))
K_target_sym = sp.simplify((K_target_ht + K_target_ht.T)/2)

# ------------- Begin LaTeX emission (explanatory write-up + formulas) -------------

say(r"% ===================== BEGIN: Analytic Discretization (Derivation) =====================")
say(r"\section*{Analytic Discretization from \((F,L)\) to \((\Phi,Q_d)\): A Step-by-Step Guide}")

# 1) Block structure (one axis)
say(r"\subsection*{1.\;Block structure (one axis)}")
say(r"""
For each Cartesian axis we stack the linear states
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
&A=\begin{bmatrix}0&0&0\\ 1&0&0\\ 0&1&0\end{bmatrix},\quad
B=\begin{bmatrix}1\\0\\0\end{bmatrix},\\
&D=-\tfrac{1}{\tau},\quad
\mathbb E[w(t)w(s)]=q_c\,\delta(t-s),\quad
q_c=\tfrac{2\sigma^2}{\tau}.
\end{aligned}
\]
Intuition:\; \(A\) is the \((v,p,S)\) kinematic chain (nilpotent: \(A^3=0\));
\(a\) is an OU process \(\dot a = -\tfrac{1}{\tau} a + w\);
noise enters only through \(a\) (the latent excitation).
""")

# 2) Transition matrix
say(r"\subsection*{2.\;Transition matrix via block-triangular exponential}")
disp_block("Block exponential formula", raw=r"""
Because \(F=\begin{bmatrix}A&B\\0&D\end{bmatrix}\) is block upper triangular,
\[
e^{F s}
=
\begin{bmatrix}
e^{A s} & \displaystyle\int_{0}^{s}\! e^{A(s-u)}\,B\,e^{D u}\,du\\[6pt]
0 & e^{D s}
\end{bmatrix},
\qquad
e^{A s}=I + A s + \tfrac{1}{2}A^2 s^2,
\quad
e^{D s}=e^{-s/\tau}.
\]
""")
disp_block("Computed e^{Fs}", E_s)
say(r"Setting \(s=h\) gives the per-axis transition matrix \(\Phi_{\text{axis}}(h)=e^{Fh}\):")
disp_block("Φ(h)", Phi)

# 3) Compact primitives and feedthroughs
say(r"\subsection*{3.\;Compact primitives (exact closed forms)}")
say(r"""
Introduce the dimensionless step \(x=\tfrac{h}{\tau}\), the OU primitives
\(\alpha=e^{-x}\), \(\mathrm{em1}=\alpha-1\) (so \(1-\alpha=-\mathrm{em1}\)),
and the kinematic monomials \(C_0=h\), \(C_1=\tfrac{h^2}{2}\), \(C_2=\tfrac{h^3}{3}\).
Then the three feedthroughs from \(a\) into \((v,p,S)\) take closed form:
\[
\phi_{va}=-\tau\,\mathrm{em1},\qquad
\phi_{pa}=\tau^2\,(x+\mathrm{em1}),\qquad
\phi_{Sa}=\tau^3\!\left(\tfrac{1}{2}x^2 - x - \mathrm{em1}\right),\qquad
\phi_{aa}=e^{-x}.
\]
These are exactly the expressions used in code via numerically stable \(\mathrm{expm1}\)-style combinations.
""")

# 4) Qd integral explanation & computed Qd
say(r"\subsection*{4.\;Discrete covariance \(Q_d\) as a noise-flow integral}")
say(r"""
The continuous-to-discrete mapping with white \(w\) is
\[
Q_d(h)=\int_{0}^{h} e^{F s}\,L\,q_c\,L^{\!\top}\,e^{F^{\!\top} s}\,ds
=\int_{0}^{h} q_c\,
\begin{bmatrix}
C_{\text{top}}(s)\\ e^{-s/\tau}
\end{bmatrix}
\begin{bmatrix}
C_{\text{top}}(s)^{\!\top} & e^{-s/\tau}
\end{bmatrix}ds,
\]
with \(C_{\text{top}}(s)=\int_{0}^{s} e^{A(s-u)}B\,e^{-u/\tau}du\).
The integrals are elementary (polynomial \(\times\) scalar exponential).
""")
disp_block("Q_d(h) (symbolic)", Qd)

# 5) Kernel K(h,τ) and primitives
say(r"\subsection*{5.\;Kernel \(K(h,\tau)\) in compact primitives}")
say(r"""
Factor \(q_c=\tfrac{2\sigma^2}{\tau}\) and define the unit kernel \(K\) by
\[
Q_{d,\text{axis}}(h)=\frac{2\sigma^2}{\tau}\;\mathrm{sym}\,K(h,\tau),
\qquad \mathrm{sym}\,K=\tfrac12(K+K^\top).
\]
Introduce additional primitives
\[
\mathrm{em1}_2=e^{-2x}-1,\quad
B_0=-\tfrac{\tau}{2}\,\mathrm{em1}_2,\quad
A_0=-\tau\,\mathrm{em1},\quad
A_1=\tau^2(-\mathrm{em1}-x\alpha),\quad
A_2=\tau^3\!\left(-2\,\mathrm{em1}+\alpha x(x+2)\right),
\]
and \(C_3=\tfrac{h^4}{4}\), \(C_4=\tfrac{h^5}{5}\).
The unit kernel before symmetrization is
\[
K(h,\tau)=
\begin{bmatrix}
K_{vv} & K_{pv} & K_{Sv} & K_{va}\\
K_{pv} & K_{pp} & K_{Sp} & K_{pa}\\
K_{Sv} & K_{Sp} & K_{SS} & K_{Sa}\\
K_{va} & K_{pa} & K_{Sa} & K_{aa}
\end{bmatrix},
\]
with entries
\[
\begin{aligned}
K_{vv}&=\tau^2\Big(C_0-2A_0+B_0\Big),\quad & K_{va}&=\tau\Big(A_0-B_0\Big),\\
K_{pp}&=\tau^2 C_2 - 2\tau^3 C_1 + 2\tau^3 A_1 + \tau^4 C_0 - 2\tau^4 A_0 + \tau^4 B_0,
& K_{pa}&=\tau A_1 + \tau^2(B_0 - A_0),\\
K_{SS}&=\tfrac{1}{4}\tau^2 C_4 - \tau^3 C_3 + 2\tau^4 C_2 - 2\tau^5 C_1 + \tau^6 C_0
        - \tau^4 A_2 + 2\tau^5 A_1 - 2\tau^6 A_0 + \tau^6 B_0,\\
K_{Sp}&=\tfrac{1}{2}\tau^2 C_3 - \tfrac{3}{2}\tau^3 C_2 + 2\tau^4 C_1 - \tau^5 C_0
        + \tfrac{1}{2}\tau^3 A_2 - 2\tau^4 A_1 + 2\tau^5 A_0 - \tau^5 B_0,\\
K_{Sv}&=\tfrac{1}{2}\tau^2(C_2 - A_2) - \tau^3(C_1 - A_1) + \tau^4(C_0 - A_0)
        - \tau^4(A_0 - B_0),\\
K_{aa}&=B_0.
\end{aligned}
\]
Finally,
\[
Q_{d,\text{axis}}(h)=\frac{2\sigma^2}{\tau}\;\mathrm{sym}\,K(h,\tau).
\]
This matches the code exactly.
""")
disp_block("K(h,τ) derived from F,L (symmetrized)", K_sym)
disp_block("Target compact K(h,τ) (primitives)", K_target_sym)

say(r"% === Entry-by-entry verification (after x=h/τ substitution) ===")
eq_print("K_{vv}", K_sym[0,0], K_target_sym[0,0])
eq_print("K_{pv}", K_sym[0,1], K_target_sym[0,1])
eq_print("K_{Sv}", K_sym[0,2], K_target_sym[0,2])
eq_print("K_{va}", K_sym[0,3], K_target_sym[0,3])
eq_print("K_{pp}", K_sym[1,1], K_target_sym[1,1])
eq_print("K_{Sp}", K_sym[1,2], K_target_sym[1,2])
eq_print("K_{pa}", K_sym[1,3], K_target_sym[1,3])
eq_print("K_{SS}", K_sym[2,2], K_target_sym[2,2])
eq_print("K_{Sa}", K_sym[2,3], K_target_sym[2,3])
eq_print("K_{aa}", K_sym[3,3], K_target_sym[3,3])

# 6) Small-step asymptotics
say(r"\subsection*{6.\;Small-step asymptotics (\(x=h/\tau \ll 1\))}")
say(r"""
For numerical robustness at high sampling rates, we use the Maclaurin expansions
\[
\alpha=1-x+\tfrac{x^2}{2}-\tfrac{x^3}{6}+\cdots,\quad
\mathrm{em1}=-x+\tfrac{x^2}{2}-\tfrac{x^3}{6}+\cdots,\quad
\mathrm{em1}_2=-2x+2x^2-\tfrac{4}{3}x^3+\cdots.
\]
Inserted into \(\phi_{\bullet a}\) and the entries of \(K\), these remove catastrophic
cancellation as \(x\to 0\). Our implementation switches branches at a small threshold.
""")

# 7) Kronecker assembly (shared τ)
say(r"\subsection*{7.\;Kronecker assembly for 3-axis shared-\(\tau\)}")
say(r"""
For the 12-state block \([v(3),p(3),S(3),a(3)]\) with shared \(\tau\) and full SPD
\(\Sigma_{aw}\in\mathbb{R}^{3\times 3}\),
\[
\boxed{
\Phi_{LL}(h)=I_3 \,\otimes\, \Phi_{\text{axis}}(h),\qquad
Q_{LL}(h)=\frac{2}{\tau}\;\Sigma_{aw}\,\otimes\,\mathrm{sym}\,K(h,\tau).
}
\]
Special cases:
\(\Sigma_{aw}=\mathrm{diag}(s_x^2,s_y^2,s_z^2)\) (anisotropic diagonal) yields block-diagonal \(Q_{LL}\)
with identical \(4\times4\) kernels scaled by \(s_x^2,s_y^2,s_z^2\).
""")

# 8) Plug into filter
say(r"\subsection*{8.\;Where this plugs into the filter}")
say(r"""
Given \(\Phi\) and \(Q_d\), the discrete prediction step is
\[
\hat{\boldsymbol{x}}_{k|k-1}=\Phi\,\hat{\boldsymbol{x}}_{k-1|k-1},\qquad
P_{k|k-1}=\Phi\,P_{k-1|k-1}\,\Phi^{\!\top}+Q_d,
\]
with the attitude block \((\delta\boldsymbol{\theta},\mathbf{b}_g)\) advanced by the
standard multiplicative update (constant \(\boldsymbol{\omega}\) over \(h\)), and the
linear block by \(\Phi_{LL},Q_{LL}\) above. All measurement updates are done in
Joseph form and we project to PSD after each update for numerical hygiene.
""")

say(r"% ===================== END: Analytic Discretization (Derivation) =====================")
