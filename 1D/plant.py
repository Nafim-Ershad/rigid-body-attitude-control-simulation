import numpy as np

class Rotational1D:
    def __init__(self, I = 1.0, tau_max = 2.0):
        self.I = I # Moment of Inertia
        self.tau_max = tau_max # Maximum torque that can be applied

    def disturbance(self, t):
        """
        A simple time-varying disturbance torque. In a real system, this could represent external forces or unmodeled dynamics.
        For example, we can use a sinusoidal disturbance to simulate periodic external forces.
        """
        
        return 0.05 * np.sin(0.5 * t)

    def dynamics(self, x, tau, t):
        _, omega = x

        tau_total = tau + self.disturbance(t)
        tau_total = np.clip(tau_total, -self.tau_max, self.tau_max)

        theta_dot = omega
        omega_dot = tau_total / self.I

        return np.array([theta_dot, omega_dot], dtype=float)