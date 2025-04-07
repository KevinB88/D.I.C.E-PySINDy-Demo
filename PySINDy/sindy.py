import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error
import pysindy as ps
import os
from matplotlib import rcParams
import shutil as sh
from tools import add_noise
from latex import generate_true_equation_latex

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

    variable_mappings = {
        "lorenz": ['x', 'y', 'z' ],
        "vanderpol": ['x', 'y'],
        "duffing": ['x', 'y'],
        "lotka": ['x', 'y'],
        "harmonic": ['x', 'y'],
        "pendulum": ["\\theta", "\\omega"],
        "linear": ['x', 'y']
    }
    symbols = variable_mappings.get(system_name, [f"x{i}" for i in range(X.shape[1])])

    csv_path = os.path.join(results_dir, f"{system_name}_trajectory.csv")
    df = pd.DataFrame(np.column_stack((t, X)), columns=["time"] + [f"x{i}" for i in range(X.shape[1])])
    df.to_csv(csv_path, index=False)
    print(f"[✓] Saved data to {csv_path}")

    # fig = plt.figure(figsize=(8, 6))
    # if X.shape[1] == 3:
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=0.5)
    #     ax.set_title(f"{system_name} Trajectory (3D)")
    #     ax.set_xlabel("x0"); ax.set_ylabel("x1"); ax.set_zlabel("x2")
    # else:
    #     for i in range(X.shape[1]):
    #         plt.plot(t, X[:, i], label=f"x{i}")
    #     plt.title(f"{system_name} Trajectories")
    #     plt.xlabel("Time"); plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, f"{system_name}_trajectory_plot.png"))
    # plt.show()

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
    if sh.which("latex"):
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
    true_eq_latex = generate_true_equation_latex(system_name, ode_args)
    for line in true_eq_latex.splitlines():
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
        eq_replaced = eq
        for j, var in enumerate(symbols):
            eq_replaced = eq_replaced.replace(f"x{j}", var)
        ax.text(0.01, y_pos, fr"${symbols[ i ]}' = {eq_replaced}$", fontsize=11, va='top', ha='left')
        y_pos -= 0.08

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{system_name}_comparison.png"), dpi=300)
    plt.close()

    X_sim = model.simulate(X[0], t)

    fig = plt.figure(figsize=(8, 6))

    if X.shape[1] == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(X[ :, 0 ], X[ :, 1 ], X[ :, 2 ], lw=1, label="True")
        ax.plot(X_sim[ :, 0 ], X_sim[ :, 1 ], X_sim[ :, 2 ], '--', lw=1, label="Predicted")
        ax.set_title(f"{system_name} Phase Space (3D)")
        ax.set_xlabel(symbols[ 0 ])
        ax.set_ylabel(symbols[ 1 ])
        ax.set_zlabel(symbols[ 2 ])
        ax.legend()
        # interactive_zoom(ax, fig)
    else:
        for i in range(X.shape[ 1 ]):
            # plt.plot(t, X[ :, i ], label=f"True {symbols[ i ]}", lw=1)
            # plt.plot(t, X_sim[ :, i ], '--', label=f"Predicted {symbols[ i ]}", lw=1)
            plt.plot(t, X[ :, i ], label=fr"True ${symbols[ i ]}$", lw=1)
            plt.plot(t, X_sim[ :, i ], '--', label=fr"Predicted ${symbols[ i ]}$", lw=1)
        plt.title(f"{system_name} Trajectories")
        plt.xlabel("Time")
        plt.legend()
        # interactive_zoom(ax, fig)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{system_name}_trajectory_comparison.png"))
    plt.show()

    mse = mean_squared_error(X, X_sim)
    print(f"[✓] MSE between true and predicted trajectories: {mse:.6f}")

    plt.figure(figsize=(12, 4))
    for i in range(X.shape[1]):
        true_vals = X[:, i]
        pred_vals = X_sim[:, i]
        # residuals = np.abs(true_vals - pred_vals)
        plt.subplot(1, X.shape[1], i + 1)
        plt.plot(t, true_vals, label="True", lw=1)
        plt.plot(t, pred_vals, '--', label="Predicted", lw=1)
        plt.fill_between(t, true_vals, pred_vals, color='gray', alpha=0.3, label="Residual")
        plt.xlabel("Time")
        # plt.ylabel(symbols[i])
        plt.ylabel(f"${symbols[ i ]}$")
        plt.legend()
    plt.suptitle(f"{system_name}: True vs SINDy Predicted\nMSE: {mse:.6f}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{system_name}_prediction_plot.png"))
    plt.show()

    mse_per_variable = [
        mean_squared_error(X[:, i], X_sim[:, i]) for i in range(X.shape[1])
    ]

    # Plot bar chart for MSE
    plt.figure(figsize=(6, 4))
    # plt.bar(range(X.shape[1]), mse_per_variable, tick_label=symbols)
    tick_labels = [ f"${s}$" for s in symbols ]
    plt.bar(range(X.shape[ 1 ]), mse_per_variable, tick_label=tick_labels)

    plt.ylabel("MSE")
    plt.title(f"{system_name}: MSE per Variable")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{system_name}_mse_bar.png"))
    plt.show()
