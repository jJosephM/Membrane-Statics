import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker
from math import pi, sqrt
from scipy.integrate import solve_bvp

# Use LaTeX to enable custom fonts
plt.rcParams["text.usetex"] = True

# Generate a synthetic "vortex" in-plane angle field Φ on a flat membrane
L = 2.5  # half-size of the square domain
N = 600  # resolution for the background field
x = np.linspace(-L + 1, L + 1, N)
y = np.linspace(-L + 1, L + 1, N)
X, Y = np.meshgrid(x, y)

# In-plane angle Φ (vortex of charge +1 at the origin)
Phi = np.arctan2(Y, X)  # radians in (-pi, pi]

# Normalize to [0, 1] for display (no explicit colormap selection per guidelines)
Phi_norm = (Phi) / (2 * np.pi)

fig, ax = plt.subplots(figsize=(5, 5))
# Map to hue in [0,1); add 0.25 for +π/2 shift (since 0.25 * 2π = π/2)
hue = -(Phi/(2*np.pi) + 0.25) % 1.0

im = ax.imshow(hue, origin='lower', extent=[-L + 1, L + 1, -L + 1, L + 1],
               cmap='hsv', interpolation='bilinear')


# # Add a colorbar (default settings, no explicit colors)
# cbar = plt.colorbar(im, ax=ax, shrink=0.9)
# cbar.set_label(r'Normalized in-plane angle $\Phi$')

# Overlay sparse magnetization arrows to show the winding (use a coarse grid)
x_step = 119
scale = 1.15
Xs = X[::x_step, ::x_step] /scale
Ys = Y[::x_step, ::x_step] /scale
print(Xs)
theta = np.arctan2(Ys, Xs)
Ux = np.cos(theta)
Uy = np.sin(theta)

ax.quiver(Xs, Ys, Ux, Uy, pivot='mid', scale=10)

# Annotations to make the figure self-contained
# ax.set_title("Flat membrane with a single in-plane vortex")
ax.set_xlabel(r"$x/\Delta_0$",size=14)
ax.set_ylabel(r"$y/\Delta_0$",size=14)

ax.axhline(
    y=0, 
    xmin=0.3, xmax=1,           # fraction of x-axis (0 = left, 1 = right)
    color="black", linestyle="--"
)

# x_start, x_end = 2.2, 2.2   # your variable positions
ax.plot([0.1, 1.25], [0.0,1.09], linestyle="-", color="black")
ax.plot([3.45,3.385], [0.1,1.09], linestyle="-", color="black")

# ax.annotate(
#     "", 
#     xy=(3.385, 1.09), xytext=(3.45, 0.1),
#     arrowprops=dict(arrowstyle="->", linestyle="-", color="black")
# )
# ax.annotate(
#     "", 
#     xy=(1.25, 1.09), xytext=(0.1, 0.0),
#     arrowprops=dict(arrowstyle="->", linestyle="-", color="black")
# )



# Mark the core and add explanatory labels
ax.plot([0], [0], marker='o', markersize=21, color='white')
ax.text(0.26, -0.21, r"Vortex core", fontsize=14, color='white')

ax.annotate(r"Winding of $\Phi$",
            xy=(-0.6, 0.65), xycoords='data',
            xytext=(-1.4, 1.6), textcoords='data',
            arrowprops=dict(arrowstyle="->",color='white'),
            fontsize=14, color='white')





## SOLVE ODE FOR DELTA AND PLOT IN INSET ##
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
print(np.size(r_sol))
print(np.size(Delta))

# Inset plot of Δ(r)
ax_in = inset_axes(ax, width="45%", height="45%", loc='upper right', borderpad=0.7)
ax_in.plot(r_sol, Delta, label=r'$\Delta(r)$')
ax_in.axhline(1.0, ls='--', label=r'$\Delta_0$',color='orange')
ax_in.axvline(0.49, ls=':', label=r'core',color='black')
ax_in.set_xlim(-0.5, 7) # Plot Δ(r) up to r=7
ax_in.set_xlabel(r'$r/\Delta_0$',size=14,color = 'white',labelpad=-1)
ax_in.set_ylabel(r'$\Delta/\Delta_0$',size=14,color = 'white',labelpad=-1)
ax_in.tick_params(colors="white")  # changes both x and y tick marks + labels

# plt.title(r'BVP solution (original sign) up to r=10')
# plt.grid(True)
ax_in.legend(loc="lower right", fontsize=14)

# ax_in.tight_layout()
ax_in.set_facecolor((0.9, 0.9, 0.99))

# ax.yaxis.set_visible(False)
factor = 2
# For x-axis
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, pos: f"{x*factor:g}")
)
# For y-axis
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda y, pos: f"{y*factor:g}")
)

plt.tight_layout()
plt.savefig(f"vortex_with_inset.png", dpi=300)
# plt.show()