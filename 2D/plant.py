import numpy as np


class Rotational2D:
    def __init__(self, Ix=1.0, Iy=1.2, tau_max=2.0):
        self.Ix = Ix
        self.Iy = Iy
        self.tau_max = tau_max

    def disturbance(self, t):
        return np.array([
            0.04 * np.sin(0.5 * t),
            0.03 * np.cos(0.7 * t)
        ], dtype=float)

    def dynamics(self, x, tau, t):
        thetaX, omegaX, thetaY, omegaY = x

        tau = np.array(tau, dtype=float) + self.disturbance(t)
        tau = np.clip(tau, -self.tau_max, self.tau_max)

        # mild coupling for learning
        couplingX = 0.02 * omegaY # effect of y-axis on x-axis
        couplingY = -0.02 * omegaX # effect of x-axis on y-axis

        thetaX_dot = omegaX
        omegaX_dot = (tau[0] + couplingX) / self.Ix # 

        thetaY_dot = omegaY
        omegaY_dot = (tau[1] + couplingY) / self.Iy

        return np.array([thetaX_dot, omegaX_dot, thetaY_dot, omegaY_dot], dtype=float)