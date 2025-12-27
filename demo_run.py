"""
DeepSwingr Demo - Pure JAX
"""
import numpy as np
from simulator import simulate_trajectory, compute_swing, optimize_seam_angle
from plotter.plt import plot_trajectory_3d

print("="*60)
print("  🏏 DeepSwingr Demo - Pure JAX Edition")
print("="*60)

velocity, release_angle, roughness, seam_angle = 35.0, 5.0, 0.8, 30.0
backend = "jaxphysics"

print(f"\n🎾 Simulating: {velocity*3.6:.0f} km/h, seam={seam_angle}°, roughness={roughness}")

# Run simulation
times, x, y, z, velocities = simulate_trajectory(velocity, release_angle, roughness, seam_angle, backend)
times, x, y, z, velocities = np.array(times), np.array(x), np.array(y), np.array(z), np.array(velocities)

# Results
valid_idx = np.where((z > 0) & (x < 20.12))[0]
last_idx = valid_idx[-1] if len(valid_idx) > 0 else len(x) - 1

print(f"\n✓ Results:")
print(f"   Pitch distance: {x[last_idx]:.2f} m")
print(f"   Lateral swing: {abs(y[last_idx] - y[0]) * 100:.2f} cm")
print(f"   Flight time: {times[last_idx]:.3f} s")
print(f"   Final speed: {np.linalg.norm(velocities[last_idx]) * 3.6:.0f} km/h")

# Save plot
fig = plot_trajectory_3d(times, x, y, z, velocities, velocity, roughness, seam_angle, use_plotly=True)
fig.write_html("trajectory_output.html")
print(f"\n📊 Saved: trajectory_output.html")

# Optimize
print(f"\n⚡ Finding optimal seam angle for outswing...")
opt_angle, max_swing = optimize_seam_angle(velocity, release_angle, roughness, "out", backend)
print(f"   Optimal: {opt_angle:.0f}° → {max_swing:.1f} cm swing")

print("\n" + "="*60)
print("  Demo Complete!")
print("="*60)

