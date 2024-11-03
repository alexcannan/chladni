"""
Chladni figure implementation based on [1].

[1]: https://www.et.byu.edu/~vps/ME505/AAEM/V10-14.pdf
"""

import numpy as np
import matplotlib.pyplot as plt


L = 1  # plate size x
M = 1  # plate size y
gamma = 0.1  # damping
v = 1  # transverse speed
s_0 = 0.1  # source amplitude
s_w = 10  # source frequency
max_sum = 5  # number of terms in the series


def mu_n(n):
    return n * np.pi / L


def lambda_m(m):
    return m * np.pi / M


def beta(n, m):
    return np.sqrt(mu_n(n)**2 * v**2 + lambda_m(m)**2 * v**2 - gamma**4 * v**4)


def phi(n, m):
    return s_0 * np.cos(mu_n(n) * (L / 2)) * np.cos(lambda_m(m) * (M / 2))


def u_n_m(n, m, t):
    return ((v**2 * phi(n, m)) / beta(n, m)) * np.sin(s_w * t) * np.exp(-gamma**2 * v**2 * t) * np.sin(beta(n, m) * t)


def u(x, y, t):
    return (4 / (L*M)) * np.sum([
        u_n_m(n, m, t) * np.cos(mu_n(n) * x) * np.cos(lambda_m(m) * y)
        for n in range(1, max_sum)
        for m in range(1, max_sum)
    ])


def frame(t):
    x = np.linspace(0, L, 100)
    y = np.linspace(0, M, 100)
    X, Y = np.meshgrid(x, y)

    U = np.vectorize(u)(X, Y, t)

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, U, levels=50, cmap='viridis')
    plt.colorbar(label="Displacement (u)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Standing Wave Displacement Gradient at t={t}")
    plt.show()


if __name__ == "__main__":
    frame(0.1)
