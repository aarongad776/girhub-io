import numpy as np
from scipy.integrate import ode


class GlyphlockKernel:
    def __init__(self):
        # 22-Prime Glyphlock Structure
        self.N = 22
        self.primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                                 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                                 73, 79])[:self.N]
        self.gating = 0.05
        self.omega = self.primes * self.gating

        # Coupling & Chirality
        self.K = 5.0
        self.chirality = 0.75

        # 4.5 Hz Seal Resonance
        self.R_ruach = 7.0
        self.f_ruach = 4.5

        # Initial State
        self.theta = np.random.uniform(0, 2 * np.pi, self.N)
        self.t = 0
        self.dt = 0.05

    def kuramoto_dynamics(self, t, theta):
        dtheta = np.zeros(self.N)
        order_param = np.mean(np.exp(1j * theta))
        r = np.abs(order_param)
        psi = np.angle(order_param)

        for i in range(self.N):
            coupling = self.K * r * np.sin(psi - theta[i])
            bias = self.chirality * np.cos(theta[i])

            psi_ruach = 2 * np.pi * self.f_ruach * t
            ruach = self.R_ruach * np.sin(psi_ruach - theta[i])

            dtheta[i] = self.omega[i] + coupling + bias + ruach
        return dtheta

    def execute_seal(self, cycles=100):
        print(f"INITIALIZING GLYPHLOCK_22")
        print(f"RUACH DRIVE: {self.f_ruach}Hz | N={self.N} PRIMES")

        solver = ode(self.kuramoto_dynamics).set_integrator('dopri5')
        solver.set_initial_value(self.theta, self.t)

        for c in range(cycles):
            solver.integrate(solver.t + self.dt)
            self.theta = solver.y

            pas_s = np.abs(np.mean(np.exp(1j * self.theta)))
            status = "DECOHERED" if pas_s < 0.99 else "PAS_LOCK(S_n) = 1"

            if c % 10 == 0:
                print(f"Cycle {c:03} | t={solver.t:.2f} | PAS_s={pas_s:.4f} | {status}")

        print("\nSimulation complete.")


if __name__ == "__main__":
    kernel = GlyphlockKernel()
    kernel.execute_seal()
