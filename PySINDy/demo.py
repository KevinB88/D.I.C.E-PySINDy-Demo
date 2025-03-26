import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import pysindy as ps
import os
from matplotlib import rcParams
import shutil

os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"


# ------------------------------------------------
# DESCRIPTION:
# This script allows for the simulation, visualization,
# and sparse regression-based discovery of differential equations
# from known dynamical systems using PySINDy.
# It handles synthetic data generation, CSV storage, optional noise,
# LaTeX rendering of true vs discovered equations, and PDF export.
# Results are stored in "results/<system_name>/" directory.
# ------------------------------------------------

# ------------------------------------------------
# Step 1: Define multiple dynamical systems
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

# ------------------------------------------------
# Step 2: True equation LaTeX dictionary (used for PDF report)
# ------------------------------------------------


true_equations_latex = {
    "lorenz": r"""
        \begin{align*}
        \dot{x} &= \sigma(y - x) \\
        \dot{y} &= x(\rho - z) - y \\
        \dot{z} &= xy - \beta z
        \end{align*}
    """,
    "vanderpol": r"""
        \begin{align*}
        \dot{x} &= y \\
        \dot{y} &= \mu(1 - x^2)y - x
        \end{align*}
    """,
    "duffing": r"""
        \begin{align*}
        \dot{x} &= y \\
        \dot{y} &= -\delta y - \alpha x - \beta x^3 + \gamma \cos(\omega t)
        \end{align*}
    """,
    "lotka": r"""
        \begin{align*}
        \dot{x} &= \alpha x - \beta x y \\
        \dot{y} &= \delta x y - \gamma y
        \end{align*}
    """,
    "harmonic": r"""
        \begin{align*}
        \dot{x} &= y \\
        \dot{y} &= -k x
        \end{align*}
    """,
    "pendulum": r"""
        \begin{align*}
        \dot{\theta} &= \omega \\
        \dot{\omega} &= -\frac{g}{L}\sin(\theta)
        \end{align*}
    """,
    "linear": r"""
        \begin{align*}
        \dot{x} &= ax + by \\
        \dot{y} &= cx + dy
        \end{align*}
    """
}

# -----------------------------------------------
# Step 3: Add noise to data (based on desired SNR)

def add_noise(X, snr_db):
    signal_power = np.mean(X ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), X.shape)
    return X + noise

# -----------------------------------------------
# Step 4: Full simulation and discovery pipeline

def run_discovery_pipeline(
    ode_func,
    ode_args=(),
    initial_conditions=None,
    t_span=(0, 20),
    t_steps=10000,
    system_name="system",
    snr_db=None,
    add_noise_to_data=False,
    sindy_threshold=0.1,
    derivative_method=None
):
    t_eval = np.linspace(t_span[0], t_span[1], t_steps)
    results_dir = os.path.join("results", system_name)
    os.makedirs(results_dir, exist_ok=True)

    sol = solve_ivp(
        fun=lambda t, y: ode_func(t, y, *ode_args),
        t_span=t_span,
        y0=initial_conditions,
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12
    )

    X_clean = sol.y.T
    t = sol.t
    X = add_noise(X_clean, snr_db) if add_noise_to_data and snr_db is not None else X_clean

    csv_path = os.path.join(results_dir, f"{system_name}_trajectory.csv")
    df = pd.DataFrame(np.column_stack((t, X)), columns=["time"] + [f"x{i}" for i in range(X.shape[1])])
    df.to_csv(csv_path, index=False)
    print(f"[✓] Saved data to {csv_path}")

    fig = plt.figure(figsize=(8, 6))
    if X.shape[1] == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=0.5)
        ax.set_title(f"{system_name} Trajectory (3D)")
        ax.set_xlabel("x0"); ax.set_ylabel("x1"); ax.set_zlabel("x2")
    else:
        for i in range(X.shape[1]):
            plt.plot(t, X[:, i], label=f"x{i}")
        plt.title(f"{system_name} Trajectories")
        plt.xlabel("Time"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{system_name}_trajectory_plot.png"))
    plt.show()

    if derivative_method is None:
        derivative_method = ps.SmoothedFiniteDifference()

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=sindy_threshold),
        differentiation_method=derivative_method
    )
    model.fit(X, t=t)
    print(f"\n[✓] Discovered equations for {system_name}:")
    model.print()

    equations_list = model.equations()

    # Ensure LaTeX is available
    os.environ[ "PATH" ] += os.pathsep + "/Library/TeX/texbin"
    if shutil.which("latex"):
        rcParams[ "text.usetex" ] = True
    else:
        print("[!] LaTeX not found, falling back to mathtext.")
        rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=(10, 4 + 0.5 * len(equations_list)))
    ax.axis("off")

    y_pos = 1.0
    ax.text(0.01, y_pos, r'\textbf{True Equations:}', fontsize=12, va='top', ha='left')
    y_pos -= 0.1

    # Parse each line of the LaTeX true equations and clean it
    for line in true_equations_latex[ system_name ].splitlines():
        stripped = line.strip()
        if (
                stripped
                and not stripped.startswith(r"\begin")
                and not stripped.startswith(r"\end")
        ):
            clean = stripped.replace("&", "").replace(r"\\", "")
            ax.text(0.01, y_pos, f"${clean}$", fontsize=11, va='top', ha='left')
            y_pos -= 0.08

    y_pos -= 0.1
    ax.text(0.01, y_pos, r'\textbf{Discovered Equations:}', fontsize=12, va='top', ha='left')
    y_pos -= 0.1

    # Print discovered equations line by line
    for i, eq in enumerate(equations_list):
        ax.text(0.01, y_pos, f"$x_{{{i}}}' = {eq}$", fontsize=11, va='top', ha='left')
        y_pos -= 0.08

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{system_name}_comparison.png"), dpi=300)
    plt.close()

    X_sim = model.simulate(X[0], t)

    plt.figure(figsize=(12, 4))
    for i in range(X.shape[1]):
        plt.subplot(1, X.shape[1], i + 1)
        plt.plot(t, X[:, i], label="True", lw=1)
        plt.plot(t, X_sim[:, i], '--', label="Predicted", lw=1)
        plt.xlabel("Time"); plt.ylabel(f"x{i}")
        plt.legend()
    plt.suptitle(f"{system_name}: True vs SINDy Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{system_name}_prediction_plot.png"))
    plt.show()

# -----------------------------------------------
# Step 5: Choose a system and run simulation


if __name__ == "__main__":
    systems = {
        "lorenz": (lorenz_system, [1.0, 1.0, 1.0], (0, 40), (10.0, 28.0, 8/3)),
        "vanderpol": (van_der_pol, [2.0, 0.0], (0, 20), (1.0,)),
        "duffing": (duffing, [1.0, 0.0], (0, 40), (0.2, -1.0, 1.0, 0.3, 1.2)),
        "lotka": (lotka_volterra, [10.0, 5.0], (0, 20), (1.5, 1.0, 1.0, 3.0)),
        "harmonic": (harmonic_oscillator, [1.0, 0.0], (0, 20), (1.0,)),
        "pendulum": (pendulum, [np.pi / 4, 0.0], (0, 20), (9.81, 1.0)),
        "linear": (linear_system, [1.0, 1.0], (0, 20), (0.5, 1.0, -1.0, 0.5))
    }

    system_name = "lorenz"
    func, ic, span, args = systems[system_name]

    run_discovery_pipeline(
        ode_func=func,
        ode_args=args,
        initial_conditions=ic,
        t_span=span,
        system_name=system_name,
        add_noise_to_data=False,
        snr_db=20,
        sindy_threshold=0.2
    )