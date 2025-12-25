"""
Demo script to show DeepSwingr in action
"""
import numpy as np
from helper.docker import get_tesseracts, cleanup_containers, PHYSICS_BACKENDS, switch_backend
from plotter.plt import plot_trajectory_3d, plot_optimization_results

print("=" * 60)
print("  DeepSwingr Demo - Cricket Ball Swing Simulation")
print("=" * 60)

# Test parameters
velocity = 35.0  # m/s (126 km/h)
release_angle = 5.0  # degrees
roughness = 0.8  # surface roughness
seam_angle = 30.0  # degrees

try:
    # Use simplephysics backend first
    backend = "simplephysics"
    url = PHYSICS_BACKENDS[backend]["url"]
    
    print(f"\n🎾 Running simulation with {backend} backend...")
    print(f"   Velocity: {velocity} m/s ({velocity*3.6:.1f} km/h)")
    print(f"   Release angle: {release_angle}°")
    print(f"   Roughness: {roughness}")
    print(f"   Seam angle: {seam_angle}°")
    
    integrator, swing, optimizer = get_tesseracts(backend_name=backend)
    
    # Run trajectory simulation
    print("\n📊 Computing trajectory...")
    res = integrator.apply({
        "initial_velocity": velocity,
        "release_angle": release_angle,
        "roughness": roughness,
        "seam_angle": seam_angle,
        "physics_url": url
    })
    
    # Extract results
    times = np.array(res["times"])
    x = np.array(res["x_positions"])
    y = np.array(res["y_positions"])
    z = np.array(res["z_positions"])
    velocities = np.array(res["velocities"])
    
    # Find pitch length (where z crosses 0 or x reaches 20.12m)
    valid_idx = np.where((z > 0) & (x < 20.12))[0]
    if len(valid_idx) > 0:
        last_idx = valid_idx[-1]
    else:
        last_idx = len(x) - 1
    
    print(f"\n✓ Trajectory Results:")
    print(f"   Pitch distance: {x[last_idx]:.2f} m")
    print(f"   Lateral swing: {abs(y[last_idx] - y[0]) * 100:.2f} cm")
    print(f"   Flight time: {times[last_idx]:.3f} s")
    print(f"   Final speed: {np.linalg.norm(velocities[last_idx]) * 3.6:.1f} km/h")
    
    # Save trajectory plot
    print("\n📈 Generating 3D plot...")
    fig = plot_trajectory_3d(times, x, y, z, velocities, velocity, roughness, seam_angle, use_plotly=True)
    fig.write_html("trajectory_output.html")
    print("   Saved to: trajectory_output.html")
    
    # Run swing calculation
    print("\n🔄 Computing swing deviation...")
    swing_res = swing.apply({
        "initial_velocity": velocity,
        "release_angle": release_angle,
        "roughness": roughness,
        "seam_angle": seam_angle,
        "physics_url": url
    })
    print(f"   Final deviation: {swing_res['final_deviation']:.2f} cm")
    
    # Run optimization
    print("\n⚡ Optimizing seam angle for maximum OUT-swing...")
    opt_res = optimizer.apply({
        "fixed_variables": {"initial_velocity": velocity, "release_angle": release_angle, "roughness": roughness},
        "optimization_variables": {"seam_angle": [-90, 90]},
        "swing_type": "out",
        "physics_url": url
    })
    print(f"   Optimal seam angle: {opt_res['optimal_parameters']['seam_angle']:.1f}°")
    print(f"   Maximum deviation: {opt_res['maximum_deviation']:.2f} cm")
    
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    
finally:
    cleanup_containers()
