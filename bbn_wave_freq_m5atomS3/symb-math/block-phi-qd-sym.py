# -*- coding: utf-8 -*-
# Derive Φ(h)=e^{Fh} and Qd(h)=∫_0^h e^{Fs} L Qc L^T e^{F^T s} ds
# from F and L for the per-axis OU chain x=[v,p,S,a]^T, and reduce to
# the SAME primitives used in the C++ code. Prints LaTeX and verifies term-wise.

import sympy as sp

# ---------------- Utilities ----------------
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

# Robust zero check with lambdify numeric fallback
def is_zero_expr(expr):
    z = sp.simplify(sp.together(sp.factor(expr)))
    if z == 0 or getattr(z, "is_zero", False):
        return True
    # numeric fallback via lambdify on the expression's free symbols
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
            # Positive generics for h, tau, sigma2; nonneg for s if it ever appears
            name = v.name
            if name in ("h",):
                subs.append(mp.mpf("0.2") + mp.mpf(random.random())*mp.mpf("1.0"))
            elif name in ("tau",):
                subs.append(mp.mpf("0.5") + mp.mpf(random.random())*mp.mpf("3.0"))
            elif name in ("sigma2",):
                subs.append(mp.mpf("0.3") + mp.mpf(random.random())*mp.mpf("2.0"))
            else:
                subs.append(mp.mpf(random.random()))
        try:
            val = f(*subs)
            valc = complex(val)  # mpmath→python complex
            if abs(valc) > 1e-9:
                return False
        except Exception:
            return False
    return True

def eq_print(name, lhs, rhs):
    ok = is_zero_expr(sp.simplify(lhs - rhs))
    tick = r"\quad(\checkmark)" if ok else r"\quad(\text{mismatch!})"
    print(r"\[ %s:\; %s \;=\; %s %s \]" %
          (name, sp.latex(sp.simplify(lhs)), sp.latex(sp.simplify(rhs)), tick))

# ---------------- Symbols ----------------
h, tau, sigma2 = sp.symbols('h tau sigma2', positive=True, real=True)
s, u = sp.symbols('s u', real=True, nonnegative=True)

# Per-axis OU chain: state x=[v,p,S,a]^T
A = sp.Matrix([[0,0,0],
               [1,0,0],
               [0,1,0]])     # nilpotent, A^3=0
B = sp.Matrix([[1],[0],[0]]) # a→dv
D = sp.Matrix([[-1/tau]])    # da/dt = -(1/τ) a + w
L = sp.Matrix([[0],[0],[0],[1]])  # noise enters a only

# ---------------- e^{F s} via block-triangular formula ----------------
I3 = sp.eye(3)
A2 = A*A

EA_s = I3 + A*s + A2*(s**2/2)        # e^{A s}
ED_s = sp.exp(-s/tau)                # e^{D s} (scalar)

TR_integrand = (I3 + A*(s-u) + A2*((s-u)**2/2)) * B * sp.exp(-u/tau)
E_TR = sp.Matrix(3,1, lambda i,j: sp.integrate(TR_integrand[i,0], (u,0,s)))

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

# ---------------- Φ(h) ----------------
Phi = sp.Matrix(E_s.subs(s, h))
disp("Φ(h) = e^{F h}", Phi)

disp("Φ-axis block entries (compact primitives)", raw=r"""
Let \(x=\frac{h}{\tau}\), \(\alpha=e^{-x}\), \(\mathrm{em1}=\alpha-1\). Then
\[
\phi_{va}=\Phi_{1,4}=-\tau\,\mathrm{em1},\quad
\phi_{pa}=\Phi_{2,4}=\tau^2\,(x+\mathrm{em1}),\quad
\phi_{Sa}=\Phi_{3,4}=\tau^3\Big(\tfrac12 x^2 - x - \mathrm{em1}\Big),\quad
\phi_{aa}=\Phi_{4,4}=\alpha.
\]
""")

# ---------------- Q_d(h) ----------------
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
disp("Q_d(h) = ∫_0^h e^{Fs} L Q_c L^T e^{F^T s} ds", Qd)

# ---------------- Reduce to primitives & verify ----------------
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

# compact targets (your C++)
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

disp("Define primitives (matching C++ code)", raw=r"""
\(x=\tfrac{h}{\tau},\;\alpha=e^{-x},\;\mathrm{em1}=\alpha-1,\;\mathrm{em1}_2=e^{-2x}-1\).
\[
A_0=-\tau\,\mathrm{em1},\;
A_1=\tau^2(-\mathrm{em1}-x\alpha),\;
A_2=\tau^3(-2\,\mathrm{em1}+\alpha x(x+2)),\;
B_0=-\tfrac{\tau}{2}\,\mathrm{em1}_2.
\]
\[
C_0=h,\;C_1=\tfrac{h^2}{2},\;C_2=\tfrac{h^3}{3},\;C_3=\tfrac{h^4}{4},\;C_4=\tfrac{h^5}{5}.
\]
""")

disp("K(h,τ) derived from F,L (symmetrized unit kernel)", K_sym)
disp("Target compact K(h,τ) in primitives", K_target)

# substitute x=h/tau into target & symmetrize
K_target_ht = sp.simplify(K_target.subs({x: h/tau}))
K_target_sym = sp.simplify((K_target_ht + K_target_ht.T)/2)

print("\n% === Entry-by-entry verification (after x=h/τ substitution) ===")
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

disp("Final discrete covariance", raw=r"""
\[
Q_{d,\text{axis}}(h)\;=\;\frac{2\sigma^2}{\tau}\;\mathrm{sym}\,K(h,\tau),
\qquad \mathrm{sym}\,K=\tfrac12\bigl(K+K^\top\bigr).
\]
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
""")
