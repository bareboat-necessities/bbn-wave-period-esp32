# phi-from-F.py
from sympy import symbols, Matrix, eye, simplify

def phi_from_F(F, h):
    """
    Return the discrete transition Phi(h) = exp(F*h) for a given continuous-time Jacobian F.
    Works symbolically or numerically (after substitution).
    """
    Phi = (F * h).exp()       # matrix exponential
    return simplify(Phi)

# --- Example (delete if you already have F) ---
if __name__ == "__main__":
    # Example: tiny 4x4 [v, p, S, a] chain Jacobian (continuous-time)
    h, tau = symbols('h tau', positive=True)
    F_axis = Matrix([
        [0, 0, 0, 1],        # vdot = a
        [1, 0, 0, 0],        # pdot = v
        [0, 1, 0, 0],        # Sdot = p
        [0, 0, 0, -1/tau]    # adot = -(1/tau) a
    ])

    Phi_axis = phi_from_F(F_axis, h)
    print("Phi_axis(h) = exp(F_axis*h):")
    print(Phi_axis)

    # If you have a full F (e.g., 21x21), just call:
    # Phi_full = phi_from_F(F_full, h)
