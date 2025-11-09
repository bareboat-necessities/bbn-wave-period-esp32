# block-phi-qd-sym.py
# Step-by-step symbolic derivation of Φ(h) = exp(F h) and
# Qd(h) = ∫_0^h e^{F s} L Qc Lᵀ e^{Fᵀ s} ds
# for the per-axis linear OU chain x = [v, p, S, a]ᵀ with
#   dv = a, dp = v, dS = p, da = -(1/τ) a + w
# L injects noise only on 'a', Qc = q_c (scalar); later we set q_c = 2 σ² / τ.

import sympy as sp

# ----------------------------- controls -----------------------------
PRINT_INTERMEDIATE_MATRICES = True   # print e^{A s}, TR(h), E_TR(s), integrand blocks, etc.
PRINT_FINAL_LATEX           = True   # also emit LaTeX (bmatrix) for Φ_axis and Qd_unit
SIMPLIFY_FINAL_FORMS        = True   # reduce to compact primitives (x, alpha, em1, em1_2)
# --------------------------------------------------------------------

# Symbols
h, s, u = sp.symbols('h s u', real=True, nonnegative=True)
tau, sigma2, qc = sp.symbols('tau sigma2 q_c', positive=True, real=True)

# ------------------ 1) Define F and L by blocks ---------------------
# State order: [v, p, S, a]
# Linear tri-chain for (v,p,S); OU for 'a'
A = sp.Matrix([[0,0,0],
               [1,0,0],
               [0,1,0]])          # nilpotent (A^3 = 0)
B = sp.Matrix([[1],[0],[0]])      # a -> v channel
D = sp.Matrix([[-1/tau]])         # OU scalar
# Assemble F = [[A, B], [0, D]] (4x4)
F = sp.Matrix([[0,0,0,1],
               [1,0,0,0],
               [0,1,0,0],
               [0,0,0,-1/tau]])

# Noise enters only 'a'
L = sp.Matrix([[0],
               [0],
               [0],
               [1]])

# Keep Qc symbolic first; later set qc = 2*sigma2/tau
Qc = sp.Matrix([[qc]])

# ------------------ 2) Compute e^{A s} (exact by nilpotency) --------
# e^{A s} = I + A s + (A^2) s^2 / 2
I3  = sp.eye(3)
A2  = A*A
eAs = I3 + A*s + A2*(s**2/2)

# ------------------ 3) Block exponential e^{F t} --------------------
# For upper triangular blocks:
# e^{F t} = [[e^{A t}, ∫_0^t e^{A(t-u)} B e^{D u} du],
#            [0,       e^{D t}]]
# We'll build both at argument 'h' (for Φ) and at generic 's' (for Qd).
eDs = sp.exp(-s/tau)            # scalar
eDh = sp.exp(-h/tau)

# 3a) Φ_TL(h) = e^{A h}
eAh = I3 + A*h + A2*(h**2/2)

# 3b) Φ_TR(h) = ∫_0^h e^{A(h-u)} B e^{D u} du.
# Change variable v = h - u for cleanliness: e^{A(h-u)} = I + A v + A² v² / 2
v = sp.symbols('v', real=True, nonnegative=True)
TR_integrand = (I3 + A*v + A2*(v**2/2)) * B * sp.exp(-(h - v)/tau)
Phi_TR = sp.Matrix(3,1, lambda i,j: sp.integrate(TR_integrand[i,j], (v,0,h)))

# 3c) Φ_BR(h) = e^{D h}
Phi_BR = eDh

# Assemble Φ(h)
Phi_axis = sp.eye(4)
Phi_axis[0:3,0:3] = eAh
Phi_axis[0:3,3:4] = Phi_TR
Phi_axis[3,3]     = Phi_BR

# ------------------ 4) Build E(s) = e^{F s} for Qd integrand --------
# E(s) has the same block form:
# E_TL(s) = e^{A s}, E_TR(s) = ∫_0^s e^{A(s-u)} B e^{D u} du, E_BR(s) = e^{D s}
eAs_s = I3 + A*s + A2*(s**2/2)
TR_s_integrand = (I3 + A*(s-u) + A2*((s-u)**2/2)) * B * sp.exp(-u/tau)
E_TR = sp.Matrix(3,1, lambda i,j: sp.integrate(TR_s_integrand[i,0], (u,0,s)))
E_BR = sp.exp(-s/tau)

# E(s) L is simply [E_TR; E_BR] because L = e_a
C_top = E_TR                 # 3x1
C_bot = sp.Matrix([[E_BR]])  # 1x1

# ------------------ 5) Construct integrand and integrate ------------
# Qd(h) = ∫_0^h E(s) L Qc Lᵀ E(s)ᵀ ds = ∫_0^h [C(s) Qc C(s)ᵀ] ds
# where C(s) = [C_top; C_bot], Qc is scalar qc.
integrand = sp.zeros(4,4)
integrand[0:3,0:3] = qc * (C_top * C_top.T)
integrand[0:3,3:4] = qc * (C_top * C_bot.T)
integrand[3:4,0:3] = qc * (C_bot * C_top.T)
integrand[3,3]     = qc * (C_bot * C_bot.T)[0,0]

Qd_unit = sp.Matrix(4,4, lambda i,j: sp.integrate(integrand[i,j], (s,0,h)))
Qd_unit = sp.simplify((Qd_unit + Qd_unit.T)/2)  # enforce symmetry

# ------------------ 6) (Optional) reduce to the C++ primitives ------
x = sp.symbols('x', positive=True, real=True)        # x = h/tau
alpha   = sp.exp(-x)                                 # e^{-x}
em1     = alpha - 1                                  # exp(-x) - 1
em1_2   = sp.exp(-2*x) - 1                           # exp(-2x) - 1

# show Φ entries reduce to code forms:
phi_va = -tau*em1
phi_pa = tau**2 * (x + em1)
phi_Sa = tau**3 * (sp.Rational(1,2)*x**2 - x - em1)
phi_aa = alpha

Phi_axis_reduced = sp.simplify(
    Phi_axis.subs({h: x*tau}).doit()
)

# Build the “K” kernel from the integrated Qd_unit when qc = 1 (unit),
# then show Qd = (2 σ² / τ) * sym(K) when qc = 2 σ² / τ.
K_from_integral = sp.simplify(Qd_unit.subs({qc: 1}).subs({h: x*tau}).doit())
Qd_from_sigma = sp.simplify(Qd_unit.subs({qc: 2*sigma2/tau}).subs({h: x*tau}).doit())

# ------------------ 7) Print results -------------------------------
if PRINT_INTERMEDIATE_MATRICES:
    print("\n# e^{A s} = I + As + A^2 s^2 / 2")
    sp.pprint(eAs)
    print("\n# Φ_TL(h) = e^{A h}")
    sp.pprint(eAh)
    print("\n# Φ_TR(h) = ∫_0^h e^{A(h-u)} B e^{D u} du")
    sp.pprint(Phi_TR)
    print("\n# Φ_BR(h) = e^{D h}")
    sp.pprint(Phi_BR)
    print("\n# Φ_axis(h) = [[e^{A h}, Φ_TR],[0, e^{D h}]]")
    sp.pprint(Phi_axis)

    print("\n# E_TR(s) = ∫_0^s e^{A(s-u)} B e^{D u} du")
    sp.pprint(E_TR)
    print("\n# E_BR(s) = e^{D s}")
    sp.pprint(E_BR)

    print("\n# integrand(s) = C(s) Qc C(s)^T (symbolic)")
    sp.pprint(integrand)
    print("\n# Qd_unit(h) = ∫_0^h integrand(s) ds (with general qc)")
    sp.pprint(Qd_unit)

print("\n# Φ_axis(h) reduced with x=h/τ → matches C++ closed forms")
sp.pprint(Phi_axis_reduced)

print("\n# K(h,τ) obtained from integral (qc=1); Qd = (2 σ² / τ) * K_sym")
sp.pprint(K_from_integral)

print("\n# Qd_axis(h) with qc = 2 σ² / τ (directly)")
sp.pprint(Qd_from_sigma)

# ------------------ 8) Optional LaTeX (clean bmatrix) ---------------
def _bmat(lx: str) -> str:
    lx = lx.replace(r"\left[\begin{matrix}", r"\begin{bmatrix}")
    lx = lx.replace(r"\end{matrix}\right]", r"\end{bmatrix}")
    lx = lx.replace(r"\left[ \begin{matrix}", r"\begin{bmatrix}")
    lx = lx.replace(r"\end{matrix} \right]", r"\end{bmatrix}")
    return lx

if PRINT_FINAL_LATEX:
    print("\n% ===== LaTeX: Φ_axis(h) (exact from F) =====")
    print(r"\[ \displaystyle " + _bmat(sp.latex(Phi_axis)) + r" \]")

    print("\n% ===== LaTeX: Qd_unit(h) = ∫ E(s) L Qc L^T E(s)^T ds (symbolic) =====")
    print(r"\[ \displaystyle " + _bmat(sp.latex(Qd_unit)) + r" \]")

    print("\n% ===== LaTeX: K(h,τ) = Qd_unit|_{qc=1, h=x τ} (for Qd = (2 σ^2 / τ) sym K) =====")
    print(r"\[ \displaystyle " + _bmat(sp.latex(K_from_integral)) + r" \]")

    print("\n% ===== LaTeX: Qd_axis(h) with qc = 2 σ^2 / τ =====")
    print(r"\[ \displaystyle " + _bmat(sp.latex(Qd_from_sigma)) + r" \]")
