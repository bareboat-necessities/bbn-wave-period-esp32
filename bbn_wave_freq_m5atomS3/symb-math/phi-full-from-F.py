# phi_full_from_F.py
from sympy import symbols, Matrix, eye, diag, exp, simplify

# -------------------------
# Symbols
# -------------------------
h, tau = symbols('h tau', positive=True, finite=True)
wx, wy, wz = symbols('w_x w_y w_z', real=True, finite=True)

# -------------------------
# Helpers
# -------------------------
def skew(wx, wy, wz):
    """[w]_x"""
    return Matrix([[0,   -wz,  wy],
                   [wz,   0,  -wx],
                   [-wy, wx,   0]])

# -------------------------
# Build F blocks
# State order: [ dtheta(3), b_g(3), v(3), p(3), S(3), a_w(3), b_a(3) ]
# -------------------------
I3 = eye(3)
Z3 = Matrix.zeros(3, 3)

# Attitude + gyro-bias (6x6): d(dtheta)/dt = -[w]_x*dtheta - I*b_g ; d(b_g)/dt = 0
W = skew(wx, wy, wz)
F_tt = -W
F_tb = -I3
F_bt = Z3
F_bb = Z3
F_AA = Matrix([[*F_tt.row(0), *F_tb.row(0)],
               [*F_tt.row(1), *F_tb.row(1)],
               [*F_tt.row(2), *F_tb.row(2)],
               [*F_bt.row(0), *F_bb.row(0)],
               [*F_bt.row(1), *F_bb.row(1)],
               [*F_bt.row(2), *F_bb.row(2)]])

# Linear OU axis (4x4) for one axis with order [v, p, S, a]:
#   vdot = a
#   pdot = v
#   Sdot = p
#   adot = -(1/tau) a
F_axis = Matrix([[0, 0, 0, 1],
                 [1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, -1/tau]])

# 3-axis linear block: blkdiag(F_axis, F_axis, F_axis)  (size 12x12)
F_LL = diag(F_axis, F_axis, F_axis)

# accel-bias block: random walk ⇒ mean dynamics zero (3x3 zeros)
F_ba = Matrix.zeros(3, 3)

# -------------------------
# Assemble full F (21x21) = blkdiag(F_AA, F_LL, F_ba)
# (No process cross-coupling between attitude and linear subsystems in your model)
# -------------------------
F_full = diag(F_AA, F_LL, F_ba)

# -------------------------
# Option A (monolithic): Phi = exp(F*h)  [symbolically heavy for 21x21]
# -------------------------
Phi_monolithic = simplify((F_full * h).exp())

# -------------------------
# Option B (blockwise exact): Phi = blkdiag(exp(F_AA*h), exp(F_LL*h), I3)
# This is mathematically identical and much faster/stabler.
# -------------------------
Phi_AA = simplify((F_AA * h).exp())
Phi_axis = simplify((F_axis * h).exp())
Phi_LL  = diag(Phi_axis, Phi_axis, Phi_axis)
Phi_blockwise = diag(Phi_AA, Phi_LL, eye(3))

# -------------------------
# Sanity: Both expressions are equal (symbolically can be expensive to prove);
# numerically, test by substitution if needed.
# -------------------------
# Example numeric substitution (optional quick check):
# subs_dict = {h: 0.01, tau: 2.3, wx: 0.02, wy: -0.01, wz: 0.05}
# print((Phi_monolithic.subs(subs_dict) - Phi_blockwise.subs(subs_dict)).n())

# Print shapes (or pretty-print LaTeX for smaller blocks)
print("F_full shape:", F_full.shape)
print("Phi (monolithic) shape:", Phi_monolithic.shape)
print("Phi (blockwise) shape:", Phi_blockwise.shape)
