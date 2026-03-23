import numpy as np


class PDController:
    def __init__(self, Kp = 0.01, Kd = 0.0, tau_max = 2.0):
        self.Kp = Kp
        self.Kd = Kd
        self.tau_max = tau_max

    def control(self, theta, omega, theta_ref):

        error = theta_ref - theta

        # PD Controller: tau = Kp * error - Kd * omega

        tau = self.Kp * error - self.Kd * omega

        return float(np.clip(tau, -self.tau_max, self.tau_max))