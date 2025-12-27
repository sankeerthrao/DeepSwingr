"""
Cricket Ball Trajectory Simulator - Pure JAX
~100x faster than containerized version!
"""

import jax
import jax.numpy as jnp
from jax import jit, grad
import diffrax as dfx
import numpy as np
from typing import Tuple
from physics import get_physics_model, analytical_physics

# Physical constants
MASS, DIAMETER, RHO_AIR, MU, G = 0.156, 0.07, 1.225, 1.5e-5, 9.81
PITCH_LENGTH = 20.12


def simulate_trajectory(
    initial_velocity: float, release_angle: float, roughness: float, seam_angle: float,
    backend: str = "jaxphysics", dt: float = 0.002, n_points: int = 500
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Simulate cricket ball trajectory. Returns times, x, y, z, velocities."""
    
    physics_fn = analytical_physics if backend == "analytical" else (
        lambda a, r, ro: get_physics_model(backend)(a, r, ro)
    )
    
    @jit
    def ball_dynamics(t, y, args):
        x, y_pos, z, vx, vy, vz = y
        v_mag = jnp.sqrt(vx**2 + vy**2 + vz**2)
        Re = jnp.clip(RHO_AIR * v_mag * DIAMETER / MU, 1e5, 1e6)
        
        forces = physics_fn(seam_angle, Re, roughness)
        v_safe = v_mag + 1e-6
        
        Fx = -forces[0] * vx / v_safe
        Fy = -forces[0] * vy / v_safe + forces[2]
        Fz = -forces[0] * vz / v_safe + forces[1] - MASS * G
        
        active = (z > 0.0) & (x < PITCH_LENGTH)
        return jnp.where(active, jnp.array([vx, vy, vz, Fx/MASS, Fy/MASS, Fz/MASS]), jnp.zeros(6))
    
    theta = jnp.deg2rad(release_angle)
    y0 = jnp.array([0.0, 0.0, 2.0, initial_velocity * jnp.cos(theta), 0.0, initial_velocity * jnp.sin(theta)])
    
    sol = dfx.diffeqsolve(
        dfx.ODETerm(ball_dynamics), dfx.Heun(), t0=0.0, t1=1.2, dt0=dt, y0=y0,
        saveat=dfx.SaveAt(ts=jnp.linspace(0, 1.2, n_points)),
        stepsize_controller=dfx.ConstantStepSize(), max_steps=n_points * 10
    )
    
    return sol.ts, sol.ys[:, 0], sol.ys[:, 1], sol.ys[:, 2], sol.ys[:, 3:6]


def compute_swing(initial_velocity: float, release_angle: float, roughness: float, 
                  seam_angle: float, backend: str = "jaxphysics") -> float:
    """Compute lateral swing in cm."""
    _, _, y, _, _ = simulate_trajectory(initial_velocity, release_angle, roughness, seam_angle, backend)
    return float((y[-1] - y[0]) * 100)


def optimize_seam_angle(initial_velocity: float, release_angle: float, roughness: float,
                        swing_type: str = "out", backend: str = "jaxphysics", 
                        n_angles: int = 37) -> Tuple[float, float]:
    """Find optimal seam angle using grid search."""
    angles = np.linspace(-90, 90, n_angles)
    best_angle, best_swing = 0.0, 0.0
    
    for angle in angles:
        swing = compute_swing(initial_velocity, release_angle, roughness, float(angle), backend)
        if (swing_type == "out" and swing > best_swing) or (swing_type == "in" and swing < -best_swing):
            best_swing = abs(swing) if swing_type == "in" else swing
            best_angle = angle
    
    return float(best_angle), float(best_swing)


if __name__ == "__main__":
    import time
    print("🏏 Testing Pure JAX Simulator\n")
    
    start = time.time()
    times, x, y, z, vel = simulate_trajectory(35.0, 5.0, 0.8, 30.0, "jaxphysics")
    print(f"First run: {(time.time()-start)*1000:.0f}ms (includes JIT)")
    
    start = time.time()
    times, x, y, z, vel = simulate_trajectory(35.0, 5.0, 0.8, 30.0, "jaxphysics")
    print(f"Second run: {(time.time()-start)*1000:.1f}ms (JIT cached)")
    print(f"Swing: {(y[-1]-y[0])*100:.2f} cm")

