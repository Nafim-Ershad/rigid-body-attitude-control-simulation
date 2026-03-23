import numpy as np


class PDController2D:
    def __init__(self, Kp=0.01, Kd=0.0, tau_max=2.0):
        self.Kp = np.array(Kp, dtype=float)
        self.Kd = np.array(Kd, dtype=float)
        self.tau_max = tau_max

    def control(self, x, x_ref):
        thetaX, omegaX, thetaY, omegaY = x
        thetaX_ref, thetaY_ref = x_ref

        error = np.array([
            thetaX_ref - thetaX,
            thetaY_ref - thetaY
        ], dtype=float)

        omega = np.array([omegaX, omegaY], dtype=float)

        tau = self.Kp * error - self.Kd * omega
        tau = np.clip(tau, -self.tau_max, self.tau_max)

        return tau