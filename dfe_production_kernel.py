import numpy as np

class DFEProductionKernel:
    def __init__(self, G_sim=6.67430e-11, c_sim=299792458.0, epsilon=0.1):
        self.G = G_sim
        self.c = c_sim
        self.epsilon_sq = epsilon ** 2
        self.gamma_local = self.G / (self.c ** 2)

    def compute_acceleration_and_potential(self, target_pos, target_idx, positions, masses):
        """
        Computes softened gravitational acceleration vectors and the positive scalar 
        potential field acting at a target coordinate point.
        Omit self-interaction dynamically when target_idx matches the index in positions loop.
        """
        accel = np.zeros(3, dtype=float)
        pot_D = 0.0
        
        for j in range(len(masses)):
            if j == target_idx:
                continue
            
            r_vec = target_pos - positions[j]
            dist_sq = np.sum(r_vec ** 2)
            dist_soft = np.sqrt(dist_sq + self.epsilon_sq)
            
            # General field definition evaluation includes all non-self indices
            pot_D += masses[j] / dist_soft
            accel += -self.G * masses[j] * r_vec / (dist_soft ** 3)
            
        return accel, pot_D

    def compute_alpha(self, pot_D, velocity):
        """
        Evaluates the first-order weak-field clock-rate proxy alpha.
        """
        phi = -self.G * pot_D
        v_sq = np.sum(velocity ** 2)
        return 1.0 + (phi / (self.c ** 2)) - (v_sq / (2.0 * self.c ** 2))

    def advance_subsystem(self, positions, velocities, masses, dt):
        """
        Non-linear state update loop using Symplectic Semi-Implicit Euler integration.
        Implements selected mechanical state registers and update maps.
        Clock-register tau advancement, tree acceleration, and finite-speed signal 
        propagation are not implemented in this module.
        """
        next_v = np.copy(velocities)
        next_x = np.copy(positions)
        
        # Step 1: Update velocities using current position states
        for i in range(len(masses)):
            accel, _ = self.compute_acceleration_and_potential(positions[i], i, positions, masses)
            next_v[i] = velocities[i] + accel * dt
            
        # Step 2: Update positions using newly computed velocity vectors (Symplectic Map)
        for i in range(len(masses)):
            next_x[i] = positions[i] + next_v[i] * dt
            
        return next_x, next_v
