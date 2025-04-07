def generate_true_equation_latex(system_name, ode_args):
    if system_name == "lorenz":
        sigma, rho, beta = ode_args
        return fr"""
        \begin{{align*}}
        \dot{{x}} &= {round(sigma,4)}(y - x) \\
        \dot{{y}} &= x({round(rho,4)} - z) - y \\
        \dot{{z}} &= xy - {round(beta,4)}z
        \end{{align*}}
        """
    elif system_name == "vanderpol":
        mu, = ode_args
        return fr"""
        \begin{{align*}}
        \dot{{x}} &= y \\
        \dot{{y}} &= {round(mu,4)}(1 - x^2)y - x
        \end{{align*}}
        """
    elif system_name == "duffing":
        delta, alpha, beta, gamma, omega = ode_args
        return fr"""
        \begin{{align*}}
        \dot{{x}} &= y \\
        \dot{{y}} &= -{round(delta,4)}y - {round(alpha,4)}x - {round(beta,4)}x^3 + 
{round(gamma,4)}\cos({round(omega,4)}t)
        \end{{align*}}
        """
    elif system_name == "lotka":
        alpha, beta, delta, gamma = ode_args
        return fr"""
        \begin{{align*}}
        \dot{{x}} &= {round(alpha,4)}x - {round(beta,4)}xy \\
        \dot{{y}} &= {round(delta,4)}xy - {round(gamma,4)}y
        \end{{align*}}
        """
    elif system_name == "harmonic":
        k, = ode_args
        return fr"""
        \begin{{align*}}
        \dot{{x}} &= y \\
        \dot{{y}} &= -{round(k,4)}x
        \end{{align*}}
        """
    elif system_name == "pendulum":
        g, L = ode_args
        return fr"""
        \begin{{align*}}
        \dot{{\theta}} &= \omega \\
        \dot{{\omega}} &= -\frac{{{round(g, 3)}}}{{{round(L, 3)}}}\sin(\theta)
        \end{{align*}}
        """

    elif system_name == "linear":
        a, b, c, d = ode_args
        return fr"""
        \begin{{align*}}
        \dot{{x}} &= {round(a,4)}x + {round(b,4)}y \\
        \dot{{y}} &= {round(c,4)}x + {round(d,4)}y
        \end{{align*}}
        """
    else:
        return r"\text{True equations not available.}"

