import numpy as np
import os
from systems import (lorenz_system, van_der_pol, duffing,
                     lotka_volterra, harmonic_oscillator, pendulum, linear_system)
from sindy import run_discovery_pipeline


# ------------------------------------------------
# DESCRIPTION:
# This script allows for the simulation, visualization,
# and sparse regression-based discovery of differential equations
# from known dynamical systems using PySINDy.
# It handles synthetic data generation, CSV storage, optional noise,
# LaTeX rendering of true vs discovered equations, and PDF export.
# Results are stored in "results/<system_name>/" directory.
# Author: Kevin Bedoya
# ------------------------------------------------

os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"

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

    system_name = "Lorenz"
    func, ic, span, args = systems[system_name]

    run_discovery_pipeline(
        ode_func=func,
        ode_args=args,
        initial_conditions=ic,
        t_span=span,
        system_name=system_name,
        add_noise_to_data=False,
        snr_db=40,
        sindy_threshold=0.2
    )
