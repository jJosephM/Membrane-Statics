import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Use LaTeX to enable custom fonts
plt.rcParams["text.usetex"] = True

R = 30
r = 25

def ode_fun(theta, y):
    """
    Computes the right-hand side of the ODE system.
    
    The state vector is:
      y[0] = Delta,
      y[1] = Delta'
      
    The ODE is given by:
      Delta'' = (T1 + p*T3 + T4 + T5) / A,
    with
      T1 = ((pi^2/(12*r^2))*(Delta')^2 - 1)/Delta^2,
      T3 = 1/Delta0^2 + (7*pi^4/(140*r^2))*(sin^2(theta)*(Delta')^2)/(R + r*sin(theta))^2
           + 1/(R + r*sin(theta))^2 + 1/r^2,
      T4 = (7*pi^4/(70*r^2))*Delta * [2*sin(theta)*Delta'/(R + r*sin(theta))^2 
           - 2*sin^2(theta)*cos(theta)*Delta'/(R + r*sin(theta))^3],
      T5 = - (pi^2/(4*r^2)) * ((1+sin^2(theta))/(R + r*sin(theta))^2)*Delta^2,
      A  = - (pi^2/(6*r^2))*(1/Delta) + (7*pi^4/(70*r^2))*(sin^2(theta)/(R + r*sin(theta))^2)*Delta.
    
    The extra parameter p multiplies T3.
    """
    Delta, dDelta = y  # Unpack the state variables
    
    # Constants
    J = 1.0
    Delta0 = 1.0
    # R = 30
    # r = 20
    
    sin = np.sin(theta)
    cos = np.cos(theta)
    R_plus = R + r * sin  # shorthand for (R + r*sin(theta))
    
    T1 = ((np.pi**2)/(12*r**2)*dDelta**2 - 1) / (Delta**2)
    T2 = ( 1 / Delta0**2 -
           1*(7*np.pi**4)/(240*r**2)*(sin**2 * dDelta**2) / (R_plus**2) +
           1/(R_plus**2) +
           1/(r**2) )
    T3 = -(7*np.pi**4)/(60*r**2) * Delta * ( (sin*cos*dDelta)/(R_plus**2) -
                                              (sin**2*cos*dDelta*r)/(R_plus**3) )
    T4 = (np.pi**2)/(4*r**2) * ((1+sin**2)/(R_plus**2)) * Delta**2
    
    A = -(np.pi**2)/(6*r**2) * (1/Delta) + (7*np.pi**4)/(60*r**2) * (sin**2/(R_plus**2)) * Delta
    
    d2Delta = (T1 + T2 + T3 + T4) / A
    return [dDelta, d2Delta]

def residuals(params):
    """
    Given a guess for the initial derivative s and the parameter p,
    integrate the ODE from 0 to 2pi and return the residuals:
      Delta(pi) - 1  and  Delta(2pi) - 1.
    """
    s, p = params
    # initial condition: Delta(0)=1, Delta'(0)=s
    y0 = [p, s]
    # Integrate with dense output enabled
    sol = solve_ivp(lambda theta, y: ode_fun(theta, y),
                    [0, 2*np.pi], y0, dense_output=True)
    # Use the dense output to evaluate at theta = pi and 2pi
    Delta_pi = sol.sol(np.pi)[0]
    Delta_2pi = sol.sol(2*np.pi)[0]
    return [(Delta_pi - p), (Delta_2pi - p)]


# Provide an initial guess for the unknowns: s and p.
initial_guess = [0.0,1.0]  # a natural guess: zero derivative and p=1
solution_params = fsolve(residuals, initial_guess)
s_sol, p_sol = solution_params
print("Found initial derivative s =", s_sol, "and parameter p =", p_sol)

# Now integrate on a fine grid for plotting using the found parameters.
theta_vals = np.linspace(0, 2*np.pi, int(2*3.141*100))
sol_full = solve_ivp(lambda theta, y: ode_fun(theta, y),
                     [0, 2*np.pi], [p_sol, s_sol], t_eval=theta_vals)

plt.figure(figsize=(5.25,3.05))
plt.plot(sol_full.t, sol_full.y[0], label=r'$\Delta(\theta)$')
plt.xlabel(r'$\theta$', size = 14)
plt.ylabel(r'$\Delta/\Delta_0$', size = 14)
plt.title(rf'$R = {R}\Delta_0,~r = {r}\Delta_0$', size = 14)
plt.tight_layout()
plt.gca().set_facecolor((0.95, 0.95, 0.99))
x = plt.gca()


# Set major ticks at multiples of pi/2
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi/2))
# Set the tick labels manually in terms of pi
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$",
                    r"$\frac{3\pi}{2}$", r"$2\pi$"],size = 14)


plt.savefig(f"Delta_sol_{R}_{r}.png", dpi=300)
# plt.show()
