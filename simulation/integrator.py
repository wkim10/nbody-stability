import numpy as np
from simulation.physics import compute_accelerations

def step(system, dt):
    pos = system.positions
    vel = system.velocities
    m = system.masses

    acc = compute_accelerations(pos, m)

    pos_new = pos + vel * dt + 0.5 * acc * dt**2  # x_(t + \Delta t) = x_t + v_t \Delta t + 1/2 a_t \Delta t^2 (second-order Taylor polynomial for x_(t + \Delta t))
    acc_new = compute_accelerations(pos_new, m)
    vel_new = vel + 0.5 * (acc + acc_new) * dt  # v_(t + \Delta t) = v_t + 1/2 (a_t + a_(t + \Delta t)) \Delta t (second-order Taylor polynomial for v_(t + \Delta t) then plugging in first-order Taylor polynomial for a_(t + \Delta t) to substitute the jerk)

    system.positions = pos_new
    system.velocities = vel_new