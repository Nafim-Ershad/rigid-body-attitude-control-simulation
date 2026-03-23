import numpy as np
import matplotlib.pyplot as plt
from integrators import rk4
from plant import Rotational2D
from controller import PDController2D


def plot_results(t, x, u, x_ref, title="2D Rotational Control"):
    thetaX = np.rad2deg(x[:, 0])
    OmegaX = np.rad2deg(x[:, 1])
    thetaY = np.rad2deg(x[:, 2])
    OmegaY = np.rad2deg(x[:, 3])

    refX = np.rad2deg(x_ref[0])
    refY = np.rad2deg(x_ref[1])

    plt.figure(figsize=(10, 8))

    plt.subplot(4, 1, 1)
    plt.plot(t, thetaX, label="thetaX")
    plt.axhline(refX, linestyle="--", label="refX")
    plt.ylabel("Angle X (deg)")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(t, OmegaX)
    plt.ylabel("Rate X (deg/s)")
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(t, thetaY, label="thetaY")
    plt.axhline(refY, linestyle="--", label="refY")
    plt.ylabel("Angle Y (deg)")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(t, OmegaY)
    plt.ylabel("Rate Y (deg/s)")
    plt.xlabel("Time (s)")
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def simulate(T=10.0, dt=0.01):
    plant = Rotational2D(Ix=1.0, Iy=1.2, tau_max=2.0)

    x_ref = np.array([
        np.deg2rad(20.0),
        np.deg2rad(-15.0)
    ], dtype=float)

    x = np.array([
        np.deg2rad(50.0), 0.0,
        np.deg2rad(-30.0), 0.0
    ], dtype=float)

    # Set Kd = [0.0, 0.0] for pure P control
    controller = PDController2D(
        Kp=[6.0, 6.0],
        Kd=[2.0, 2.0],
        tau_max=plant.tau_max
    )

    t_hist = []
    x_hist = []
    u_hist = []

    t = 0.0
    while t <= T:
        tau = controller.control(x, x_ref)

        def f(xi, ti):
            return plant.dynamics(xi, tau, ti)

        x = rk4(f, x, t, dt)

        t_hist.append(t)
        x_hist.append(x.copy())
        u_hist.append(tau.copy())

        t += dt

    return np.array(t_hist), np.array(x_hist), np.array(u_hist), x_ref


if __name__ == "__main__":
    t, x, u, ref = simulate()
    plot_results(t, x, u, ref, "2D Rotational Control")