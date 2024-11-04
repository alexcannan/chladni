from functools import lru_cache
import math

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from numba import cuda
import numpy as np


L = 1
M = 1
gamma = 0.1
v = 1
s_0 = 0.2
s_w = 54
max_sum = 60
resolution = 300


@lru_cache
def mu_n(n):
    return n * np.pi / L


@lru_cache
def lambda_m(m):
    return m * np.pi / M


@lru_cache
def beta(n, m):
    return np.sqrt(mu_n(n)**2 * v**2 + lambda_m(m)**2 * v**2 - gamma**4 * v**4)


@lru_cache
def phi(n, m):
    return s_0 * np.cos(mu_n(n) * (L / 2)) * np.cos(lambda_m(m) * (M / 2))


@cuda.jit
def compute_u(U, t, x_vals, y_vals, mu_n_vals, lambda_m_vals, beta_vals, phi_vals):
    i, j = cuda.grid(2)
    if i < U.shape[0] and j < U.shape[1]:
        x = x_vals[i]
        y = y_vals[j]
        u_sum = 0.0
        for n in range(1, max_sum):
            for m in range(1, max_sum):
                beta_nm = beta_vals[n, m]
                phi_nm = phi_vals[n, m]
                term = ((v**2 * phi_nm) / beta_nm) * np.sin(s_w * t) * math.exp(-gamma**2 * v**2 * t)
                term *= np.sin(beta_nm * t) * np.cos(mu_n_vals[n] * x) * np.cos(lambda_m_vals[m] * y)
                u_sum += term
        U[i, j] = (4 / (L * M)) * u_sum


def frame(t):
    x = np.linspace(0, L, resolution)
    y = np.linspace(0, M, resolution)
    X, Y = np.meshgrid(x, y)

    U = np.zeros((resolution, resolution), dtype=np.float32)

    mu_n_vals = np.array([mu_n(n) for n in range(max_sum)], dtype=np.float32)
    lambda_m_vals = np.array([lambda_m(m) for m in range(max_sum)], dtype=np.float32)
    beta_vals = np.array([[beta(n, m) for m in range(max_sum)] for n in range(max_sum)], dtype=np.float32)
    phi_vals = np.array([[phi(n, m) for m in range(max_sum)] for n in range(max_sum)], dtype=np.float32)

    U_device = cuda.to_device(U)
    x_vals_device = cuda.to_device(x)
    y_vals_device = cuda.to_device(y)
    mu_n_vals_device = cuda.to_device(mu_n_vals)
    lambda_m_vals_device = cuda.to_device(lambda_m_vals)
    beta_vals_device = cuda.to_device(beta_vals)
    phi_vals_device = cuda.to_device(phi_vals)

    threadsperblock = (16, 16)
    blockspergrid_x = (U.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (U.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    compute_u[(blockspergrid_x, blockspergrid_y), threadsperblock](
        U_device, t, x_vals_device, y_vals_device, mu_n_vals_device, lambda_m_vals_device, beta_vals_device, phi_vals_device
    )

    U = U_device.copy_to_host()

    plt.figure(figsize=(8, 6))
    contourf_plot = plt.contourf(X, Y, U, levels=50, cmap='viridis')
    plt.colorbar(contourf_plot, label="Displacement (u)")
    _zero_crossings = plt.contour(X, Y, U, levels=[0], colors='white')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Standing Wave Displacement Gradient at t={t}")
    plt.show()


def anim():
    x = np.linspace(0, L, resolution)
    y = np.linspace(0, M, resolution)
    X, Y = np.meshgrid(x, y)

    U = np.zeros((resolution, resolution), dtype=np.float32)

    mu_n_vals = np.array([mu_n(n) for n in range(max_sum)], dtype=np.float32)
    lambda_m_vals = np.array([lambda_m(m) for m in range(max_sum)], dtype=np.float32)
    beta_vals = np.array([[beta(n, m) for m in range(max_sum)] for n in range(max_sum)], dtype=np.float32)
    phi_vals = np.array([[phi(n, m) for m in range(max_sum)] for n in range(max_sum)], dtype=np.float32)

    U_device = cuda.to_device(U)
    x_vals_device = cuda.to_device(x)
    y_vals_device = cuda.to_device(y)
    mu_n_vals_device = cuda.to_device(mu_n_vals)
    lambda_m_vals_device = cuda.to_device(lambda_m_vals)
    beta_vals_device = cuda.to_device(beta_vals)
    phi_vals_device = cuda.to_device(phi_vals)

    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    ax.axis("off")
    U = np.zeros((resolution, resolution), dtype=np.float32)
    _contour = ax.contourf(X, Y, U, levels=50, cmap="viridis")
    _zero_crossings = ax.contour(X, Y, U, levels=[0], colors="white")


    def update(t):
        print(t)
        threadsperblock = (8, 8)
        blockspergrid_x = (U.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (U.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        compute_u[(blockspergrid_x, blockspergrid_y), threadsperblock](
            U_device, t, x_vals_device, y_vals_device, mu_n_vals_device, lambda_m_vals_device, beta_vals_device, phi_vals_device
        )

        # Copy result back to host
        U[:] = U_device.copy_to_host()

        ax.clear()
        contour = ax.contourf(X, Y, U, levels=50, cmap="viridis")
        _zero_crossings = ax.contour(X, Y, U, levels=[0], colors="white")
        # print(U.shape, np.max(U))
        return [contour]

    # Create the animation
    anim = FuncAnimation(fig, update, frames=np.linspace(0, 20, 2001), blit=True)
    anim.save("wave.mp4", fps=15, writer="ffmpeg")


if __name__ == "__main__":
    # frame(1000)
    anim()