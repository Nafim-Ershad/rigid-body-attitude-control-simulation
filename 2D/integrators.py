def rk4(func, x, t, dt):
    k1 = func(x, t)
    k2 = func(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = func(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = func(x + dt * k3, t + dt)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)