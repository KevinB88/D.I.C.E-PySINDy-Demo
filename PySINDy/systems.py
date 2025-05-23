import numpy as np
# Each system returns the derivative given current state and parameters

# 1. Lorenz System:
# Models atmospheric convection with three variables (x, y, z)
# Parameters: sigma, rho, beta
# Typical chaotic system

def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8 / 3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# 2. Van der Pol Oscillator:
# Nonlinear oscillator with damping that depends on amplitude
# Parameters: mu (nonlinearity and strength of damping)

def van_der_pol(t, state, mu=1.0):
    x, y = state
    dx = y
    dy = mu * (1 - x**2) * y - x
    return [dx, dy]

# 3. Duffing Oscillator:
# Models a driven damped nonlinear oscillator
# Parameters: delta, alpha, beta, gamma, omega

def duffing(t, state, delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.2):
    x, y = state
    dx = y
    dy = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
    return [dx, dy]

# 4. Lotka-Volterra Equations:
# Models predator-prey population dynamics
# Parameters: alpha, beta, delta, gamma

def lotka_volterra(t, state, alpha=1.5, beta=1.0, delta=1.0, gamma=3.0):
    x, y = state
    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y
    return [dx, dy]

# 5. Harmonic Oscillator:
# Classic linear spring-mass system
# Parameters: k (spring constant)

def harmonic_oscillator(t, state, k=1.0):
    x, y = state
    dx = y
    dy = -k * x
    return [dx, dy]

# 6. Simple Pendulum:
# Models angular displacement and velocity of a pendulum
# Parameters: g (gravity), L (length)

def pendulum(t, state, g=9.81, L=1.0):
    theta, omega = state
    dtheta = omega
    domega = -(g / L) * np.sin(theta)
    return [dtheta, domega]

# 7. Linear System:
# 2D linear system with matrix-defined behavior
# Parameters: a, b, c, d (matrix elements)

def linear_system(t, state, a=0.5, b=1.0, c=-1.0, d=0.5):
    x, y = state
    dx = a * x + b * y
    dy = c * x + d * y
    return [dx, dy]
