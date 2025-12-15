"""
Cricket Ball Trajectory Simulator
Simulates ball flight over 22 yards using trained neural network for aerodynamics.
"""
import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract


def simulate_trajectory(
    initial_velocity: float,
    release_angle: float,
    roughness: float,
    seam_angle: float,
    dt: float = 0.001,
    pitch_length: float = 20.12,
    debug: bool = False
):
    """Simulate cricket ball trajectory using simplephysics tesseract for aerodynamic forces."""

    v0 = initial_velocity
    theta = jnp.deg2rad(release_angle)

    vx = v0 * jnp.cos(theta)
    vy = 0.0
    vz = v0 * jnp.sin(theta)

    x, y, z = 0.0, 0.0, 2.0

    mass = 0.156
    diameter = 0.07
    rho_air = 1.225
    mu = 1.5e-5
    g = 9.81

    positions = [[x, y, z]]
    velocities = [[vx, vy, vz]]
    times = [0.0]

    t = 0.0
    max_time = 5.0
    step_count = 0

    # Connect to pre-started simplephysics via URL
    # Using the network alias 'simplephysics' on port 8000
    physics = Tesseract.from_url("http://simplephysics:8000")

    while x < pitch_length and z > 0 and t < max_time:
        v_mag = jnp.sqrt(vx**2 + vy**2 + vz**2)
        Re = rho_air * v_mag * diameter / mu
        Re = jnp.clip(Re, 1e5, 1e6)

        # Get forces from simplephysics tesseract
        forces = physics.apply({
            "notch_angle": seam_angle,
            "reynolds_number": float(Re),
            "roughness": roughness
        })['force_vector']

        if debug and step_count < 3:
            print(f"\nStep {step_count}:")
            print(f"  Velocity: {v_mag:.2f} m/s ({v_mag*3.6:.1f} km/h)")
            print(f"  Reynolds: {Re:.2e}")
            print(
                f"  Forces: Drag={forces[0]:.4f}N, Lift={forces[1]:.4f}N, Side={forces[2]:.4f}N")

        if v_mag > 1e-6:
            vx_norm = vx / v_mag
            vy_norm = vy / v_mag
            vz_norm = vz / v_mag
        else:
            break

        F_drag_x = -forces[0] * vx_norm
        F_drag_y = -forces[0] * vy_norm
        F_drag_z = -forces[0] * vz_norm

        F_lift_z = forces[1]
        F_swing_y = forces[2]

        Fx = F_drag_x
        Fy = F_drag_y + F_swing_y
        Fz = F_drag_z + F_lift_z - mass * g

        if debug and step_count < 3:
            print(
                f"  Applied Forces: Fx={Fx:.4f}N, Fy={Fy:.4f}N, Fz={Fz:.4f}N")
            print(f"  Gravity: {-mass*g:.4f}N")

        ax = Fx / mass
        ay = Fy / mass
        az = Fz / mass

        vx = vx + ax * dt
        vy = vy + ay * dt
        vz = vz + az * dt

        x = x + vx * dt
        y = y + vy * dt
        z = z + vz * dt

        t = t + dt
        step_count += 1

        if step_count % 10 == 0:
            positions.append([float(x), float(y), float(z)])
            velocities.append([float(vx), float(vy), float(vz)])
            times.append(float(t))

    if debug:
        print(f"\nFinal statistics:")
        print(f"  Total steps: {step_count}")
        print(f"  Flight time: {t:.3f}s")
        print(f"  Distance: {x:.2f}m")
        print(f"  Lateral deviation: {abs(y)*100:.2f}cm")
        print(f"  Final position: x={x:.2f}, y={y:.4f}, z={z:.2f}")

    positions = np.array(positions)
    velocities = np.array(velocities)
    times = np.array(times)

    if debug:
        print(
            f"  Y range: min={positions[:, 1].min():.4f}, max={positions[:, 1].max():.4f}")

    return times, positions[:, 0], positions[:, 1], positions[:, 2], velocities
