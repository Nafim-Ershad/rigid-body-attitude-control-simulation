import numpy as np
import matplotlib.pyplot as plt
from integrators import rk4
from plant import Rotational1D
from controller import PDController


def plot_results(t, x, u, theta_ref, title="1D Rotational Control"):
    theta = np.rad2deg(x[:, 0])
    omega = np.rad2deg(x[:, 1])
    theta_ref_deg = np.rad2deg(theta_ref)

    plt.figure(figsize=(10, 7))

    plt.subplot(3, 1, 1)
    plt.plot(t, theta, label="theta")
    plt.axhline(theta_ref_deg, linestyle="--", label="reference")
    plt.ylabel("Angle (deg)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, omega)
    plt.ylabel("Angular rate (deg/s)")
    plt.xlabel("Time (s)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, u)
    plt.ylabel("Torque")
    plt.xlabel("Time (s)")
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def simulate(T, dt):
    plant = Rotational1D(I=1.0, tau_max=2.0)

    theta_ref = np.deg2rad(30.0)
    x = np.array([np.deg2rad(60.0), 0.0], dtype=float)

    controller = PDController(Kp=8.0, Kd=2.5, tau_max=plant.tau_max)

    t_hist = []
    x_hist = []
    u_hist = []

    t = 0.0
    while t <= T:
        tau = controller.control(x[0], x[1], theta_ref)

        def f(xi, ti):
            return plant.dynamics(xi, tau, ti)

        x = rk4(f, x, t, dt)

        t_hist.append(t)
        x_hist.append(x.copy())
        u_hist.append(tau)

        t += dt

    return np.array(t_hist), np.array(x_hist), np.array(u_hist), theta_ref


if __name__ == "__main__":
    t, x, u, ref = simulate(T=10.0, dt=0.01)
    plot_results(t, x, u, ref, "1D Rotational Control")