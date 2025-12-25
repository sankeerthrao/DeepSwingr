"""
Demo script for JAX-CFD physics backend
"""
import numpy as np
from helper.docker import get_tesseracts, cleanup_containers, PHYSICS_BACKENDS
from plotter.plt import plot_trajectory_3d

print("=" * 60)
print("  DeepSwingr Demo - JAX-CFD Physics Backend")
print("=" * 60)

# Test parameters
velocity = 35.0
release_angle = 5.0
roughness = 0.8
seam_angle = 30.0

try:
    backend = "jaxphysics"
    url = PHYSICS_BACKENDS[backend]["url"]
    
    print(f"\n🎾 Running simulation with {backend} backend...")
    print(f"   (Using actual Navier-Stokes CFD solver!)")
    
    integrator, swing, optimizer = get_tesseracts(backend_name=backend)
    
    print("\n📊 Computing trajectory...")
    res = integrator.apply({
        "initial_velocity": velocity,
        "release_angle": release_angle,
        "roughness": roughness,
        "seam_angle": seam_angle,
        "physics_url": url
    })
    
    times = np.array(res["times"])
    x = np.array(res["x_positions"])
    y = np.array(res["y_positions"])
    z = np.array(res["z_positions"])
    velocities = np.array(res["velocities"])
    
    valid_idx = np.where((z > 0) & (x < 20.12))[0]
    last_idx = valid_idx[-1] if len(valid_idx) > 0 else len(x) - 1
    
    print(f"\n✓ JAX-CFD Trajectory Results:")
    print(f"   Pitch distance: {x[last_idx]:.2f} m")
    print(f"   Lateral swing: {abs(y[last_idx] - y[0]) * 100:.2f} cm")
    print(f"   Flight time: {times[last_idx]:.3f} s")
    print(f"   Final speed: {np.linalg.norm(velocities[last_idx]) * 3.6:.1f} km/h")
    
    # Save trajectory plot
    fig = plot_trajectory_3d(times, x, y, z, velocities, velocity, roughness, seam_angle, use_plotly=True)
    fig.write_html("trajectory_jax.html")
    print("\n📈 Saved JAX-CFD trajectory to: trajectory_jax.html")
    
    # Run optimization
    print("\n⚡ Optimizing seam angle for maximum OUT-swing (JAX-CFD)...")
    opt_res = optimizer.apply({
        "fixed_variables": {"initial_velocity": velocity, "release_angle": release_angle, "roughness": roughness},
        "optimization_variables": {"seam_angle": [-90, 90]},
        "swing_type": "out",
        "physics_url": url
    })
    print(f"   Optimal seam angle: {opt_res['optimal_parameters']['seam_angle']:.1f}°")
    print(f"   Maximum deviation: {opt_res['maximum_deviation']:.2f} cm")
    
    print("\n" + "=" * 60)
    print("  JAX-CFD Demo Complete!")
    print("=" * 60)
    
finally:
    cleanup_containers()
