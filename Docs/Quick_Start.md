# PySINDy Equation Discovery: Quickstart Guide

## How to Run 

From the root directory of your project, execute the script as a module to ensure relative imports work correctly:

```python
python -m PySINDy.main
```

This command triggers a simulation of a specified system, runs PySINDy to discover governing equations, and stores the results in a dedicated results/<system_name>/ folder.

##  Example Usage

```python
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
```

## Customize and Run
- Change the system by modifying system_name (e.g., "duffing", "pendulum")
- Add noise by setting add_noise_to_data=True and tweaking snr_db
- Control sparsity with the sindy_threshold parameter

## Output Description

Once execution completes, results are saved in:
results/<system_name>/

The output includes:

- simulation_plot.png: Plot of simulated system trajectory
- comparison_plot.png: Overlay of true vs discovered model
- discovered_equation.pdf: LaTeX-rendered equations side-by-side
- simulation_data.csv: Time-series data of the simulation

These outputs help validate the accuracy and interpretability of the discovered equations.