import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt
from scipy.integrate import solve_bvp

# Use LaTeX to enable custom fonts
plt.rcParams["text.usetex"] = True

# Parameters
n = 1.0        # topological charge
r0 = 1e-5      # small cutoff radius near the origin
R  = 4000.0      # large outer radius for boundary condition

# Inner slope constant: Δ(r) ~ c r near r=0
c = 1.0 / np.sqrt(n**2 - pi**2/12.0)

# ODE system (original sign):
# Variables: w = ln Δ, u = r w' = r Δ'/Δ
def fun(r, Y):
    w, u = Y
    dw = u / r
    du = (6.0/pi**2) * (r * (1.0 - np.exp(-2*w)) + n**2 / r) - 0.5 * (u**2 / r)
    return np.vstack((dw, du))

# Boundary conditions:
# Inner: w(r0) = ln(c r0)
# Outer: u(R) = n^2 / R^2   (far-field asymptotic condition)
def bc(Y0, Y1):
    w0, u0 = Y0
    w1, u1 = Y1
    return np.array([w0 - np.log(c*r0), u1 - n**2 / R**2])

# Discretization: logarithmic mesh from r0 to R
r = np.geomspace(r0, R, 800)

# Initial guess for (w,u):
t = (np.log(r) - np.log(r0)) / (np.log(R) - np.log(r0))
w_guess = (1 - t) * np.log(c * r) + t * 0.0   # interpolate from core to flat
u_guess = (1 - t) * 1.0 + t * (n**2 / R**2)   # interpolate from 1 to asymptotic value
Y_guess = np.vstack((w_guess, u_guess))

# Solve BVP
sol = solve_bvp(fun, bc, r, Y_guess, tol=1e-4, max_nodes=200000)
print("Solver status:", sol.status, sol.message)

# Extract solution
r_sol = sol.x
Delta = np.exp(sol.y[0])

Dp = (Delta[1]-Delta[0])/(r_sol[1]-r_sol[0])

print("D prime is = ", Dp)

# Plot Δ(r) up to r=10
plt.figure(figsize=(4,4))
plt.plot(r_sol, Delta, label=r'$\Delta(r)$')
plt.plot(r_sol, -1.0/(r_sol+1)**3 + 1, label=r'$-\frac{1}{(r+1)^2} + 1$')
# plt.plot(r_sol, -1.0/(r_sol+1)**2, label=r'$-\frac{1}{(r+1)^2}$')
plt.plot(r_sol, 2.373*r_sol, label=r'$\alpha r$')
plt.axhline(1.0, ls='--', label=r'$\Delta_0$',color='orange')
plt.xlim(0.01, 7.0)
plt.ylim(0.01, 1.0)
plt.xlabel(r'$r/\Delta_0$',size=13)
plt.ylabel(r'$\Delta/\Delta_0$',size=13)
# plt.title(r'BVP solution (original sign) up to r=10')
# plt.grid(True)
plt.legend()
plt.tight_layout()
plt.gca().set_facecolor((0.95, 0.95, 0.99))

plt.show()
