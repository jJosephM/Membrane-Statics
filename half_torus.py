import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Use LaTeX to enable custom fonts
plt.rcParams["text.usetex"] = True

# Torus parameters
R = 2.0   # major radius
r = 1.0   # minor radius

# Parameter ranges
phi = np.linspace(5*np.pi/18, 3*np.pi/2 + np.pi/12 + 5*np.pi/18, 400)       # azimuth (toroidal) angle, half torus
theta = np.linspace(0, 2*np.pi, 400)   # poloidal angle
Phi, Theta = np.meshgrid(phi, theta, indexing='xy')

# Torus embedding (parametric equations)
X = (R + r*np.cos(Theta)) * np.cos(Phi)
Y = (R + r*np.cos(Theta)) * np.sin(Phi)
Z = r * np.sin(Theta)

# Color mapping: Φ_total = φ + θ, wrap to [0, 2π)
Phi_total = (Phi + Theta) % (2*np.pi)
hue = Phi_total / (2*np.pi)

# vals = [0.25,0.5,0.75,0.99]
# for i in vals:
#     for j in vals:  

# Lower saturation and brightness for "matte" colors
sat = 0.9 # Saturation is like a haziness
val = 0.8 # value is like a brightness emitted from the surface and then of course there are also other parameters but idk what they are
HSV = np.stack([hue, np.ones_like(hue)*sat, np.ones_like(hue)*val], axis=-1)
RGB = hsv_to_rgb(HSV)


# Plotting
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, facecolors=RGB, rstride=1, cstride=1, cmap='viridis',
                antialiased=False, linewidth=0, shade=False)
ax.plot([0,0],[0,0],[-r,r], linestyle="-", color="black")



# Emphasize cut edges at φ=0 and φ=π
# ax.plot(X[0, :], Y[0, :], Z[0, :], color='black', lw=50)
# ax.plot(X[-1, :], Y[-1, :], Z[-1, :], color='black', lw=50)

# Aesthetics
width = 2*(R+r)
height = 2*r
ax.set_box_aspect((width, width, height))
# ax.set_title(r'Half torus colored by $\Phi=\phi+\theta$')
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.view_init(elev=25, azim=0)  # adjust view angle

plt.tight_layout()
plt.savefig(f"skewed_half_torus_{sat}_{val}.png", dpi=300)

# plt.show()
