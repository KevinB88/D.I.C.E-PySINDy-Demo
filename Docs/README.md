# PySINDy Differential Equation Discovery Pipeline

## Summary

This script provides a complete pipeline for simulating, visualizing, and discovering governing differential equations using **PySINDy** (Sparse Identification of Nonlinear Dynamical systems). The script supports multiple well-known dynamical systems, generates synthetic data, and applies sparse regression techniques to infer the underlying mathematical models.

###  Key Features

- Simulation of well-known systems (e.g., Lorenz, Van der Pol, Duffing, Pendulum)
- Storage of results (trajectories, plots, and equations) to `results/<system_name>/`
- Optional noise injection for robust testing
- Mean Squared Error (MSE) calculation between ground truth and discovered trajectories
- Side-by-side LaTeX rendering of **true** vs **discovered** equations
- PDF export of comparison and results

> **Author:** Kevin Bedoya

---

##  Dependencies

Ensure the following libraries are installed before running the script:

```python
import numpy as np                       # For numerical computations and initial conditions
import matplotlib.pyplot as plt         # For plotting simulation and discovery results
import pandas as pd                     # For structured CSV I/O
from scipy.integrate import solve_ivp   # For solving ODEs over a time interval
from sklearn.metrics import mean_squared_error  # For error quantification
import pysindy as ps                    # Core PySINDy library for sparse regression
import os                               # Filesystem utilities
from matplotlib import rcParams         # Plot configuration for LaTeX rendering
import shutil as sh                     # Directory/file cleanup and copy operations
