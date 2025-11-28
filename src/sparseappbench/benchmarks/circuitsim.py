"""
Name: Circuit Simulation Solver
Author: Akarsh Duddu
Email: aduddu3@gatech.edu
"""

from collections.abc import Callable
from functools import partial

import numpy as np

import matplotlib.pyplot as plt


def solve_ivp2(
    dydx: Callable[[float, float], float], span: tuple[float, float], y0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.linspace(span[0], span[1], 1000)
    step = inputs[1] - inputs[0]
    outputs = np.zeros((len(inputs), len(y0)))
    outputs[0] = y0

    for i in range(1, inputs.shape[0]):
        # y_new = y + dy/dx * delta x
        outputs[i] = (
            outputs[i - 1] + np.array(dydx(inputs[i - 1], outputs[i - 1])) * step
        )

    return (inputs, outputs)


def rlc(t, state, R, L, C, Vs):
    Vc = state[0]
    dVc = state[1]  # dx1/dt = x2

    Vin = Vs(t)

    d2Vc = (Vin - Vc - R * C * dVc) / (L * C)  # dx2/dt = (Vin - x1 - RCx2) / LC
    return (dVc, d2Vc)


def lotka_volterra(t, y_t, a, b, c, d):
    x, y = y_t
    dxdt = a * x - b * x * y
    dydt = d * x * y - c * y
    return (dxdt, dydt)


def rc_ode(t, Vc, R, C, Vs_func):
    """
    The right-hand side of the first-order RC differential equation.
    t: current time
    Vc: current capacitor voltage
    R, C: circuit constants
    Vs_func: function for the input source voltage V_S(t)
    """
    tau = R * C
    Vs = Vs_func(t)  # Get the current source voltage (allows for any input!)
    return (Vs - Vc) / tau


def step_input(t):
    """A simple 5V step input starting at t=0."""
    return 5.0 if t >= 0 else 0.0


def display_rc():
    R = 10e3  # 10 kΩ
    C = 1e-6  # 1 µF
    t_max = 5 * R * C
    V_C_initial = 0.0

    dVdt = partial(rc_ode, R=R, C=C, Vs_func=step_input)
    # dVdt = lambda t, Vc: rc_ode(t, Vc, R, C, step_input)

    time, voltage = solve_ivp2(dVdt, (0, t_max), [V_C_initial])

    plt.figure(figsize=(8, 5))
    plt.plot(time * 1e3, voltage, label="Numerical Solution (RK4)")
    plt.title("RC Circuit Charging - Numerical Solution")
    plt.xlabel("Time (ms)")
    plt.ylabel("Capacitor Voltage (V)")
    plt.grid(True)
    plt.legend()
    plt.show()


def display_lv():
    # Parameters
    a = 1.1  # prey birth rate
    b = 0.4  # predation rate
    c = 0.4  # predator death rate
    d = 0.1  # predator reproduction rate

    # Initial populations
    y0 = [10, 5]

    # ODE wrapper
    dydt = partial(lotka_volterra, a=a, b=b, c=c, d=d)
    # dydt = lambda t, y: lotka_volterra(t, y, a, b, c, d)

    # Solve
    t, sol = solve_ivp2(dydt, (0, 30), y0)
    # print(t)
    print(sol)
    prey = sol[:, 0]
    predators = sol[:, 1]

    # Plot time series (single chart, multiple lines allowed, no colors specified)
    plt.figure(figsize=(10, 6))
    plt.plot(t, prey, label="Prey")
    plt.plot(t, predators, label="Predators")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Lotka-Volterra Predator-Prey Dynamics")
    plt.grid(True)
    plt.legend()
    plt.show()


def display_rlc():
    R = 100
    L = 10e-3
    C = 1e-6
    t_max = 100 * R * C

    y0 = np.array([0.0, 0.0])
    dVdt = partial(rlc, R=R, L=L, C=C, Vs=step_input)
    # dVdt = lambda t, y: rlc(t, y, R, L, C, step_input)
    t, sol = solve_ivp2(dVdt, (0, t_max), y0)
    print(sol.shape)

    plt.plot(t * 1000, sol[:, 0])
    plt.xlabel("Time (ms)")
    plt.ylabel("Capacitor Voltage (V)")
    plt.title("Series RLC Step Response (Using solve_ivp2)")
    plt.ylim((4, 6))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    display_rlc()
