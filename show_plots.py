"""Generate comparison plots showing swing clearly"""
import numpy as np
import matplotlib.pyplot as plt
from helper.docker import get_tesseracts, cleanup_containers, PHYSICS_BACKENDS

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

try:
    # Run both backends
    for idx, backend in enumerate(["simplephysics", "jaxphysics"]):
        url = PHYSICS_BACKENDS[backend]["url"]
        integrator, _, _ = get_tesseracts(backend_name=backend)
        
        res = integrator.apply({
            "initial_velocity": 35.0,
            "release_angle": 5.0,
            "roughness": 0.8,
            "seam_angle": 30.0,
            "physics_url": url
        })
        
        t = np.array(res["times"])
        x = np.array(res["x_positions"])
        y = np.array(res["y_positions"]) 
        z = np.array(res["z_positions"])
        v = np.array(res["velocities"])
        
        # Top view (swing)
        ax1 = axes[idx, 0]
        ax1.plot(x, y * 100, 'b-', linewidth=2.5)
        ax1.scatter(x[0], y[0] * 100, c='green', s=150, marker='o', zorder=5, label='Release')
        ax1.scatter(x[-1], y[-1] * 100, c='red', s=150, marker='X', zorder=5, label='Landing')
        ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax1.fill_between(x, 0, y * 100, alpha=0.3, color='blue')
        ax1.set_xlabel('Distance (m)', fontsize=12)
        ax1.set_ylabel('Lateral Swing (cm)', fontsize=12)
        ax1.set_title(f'{backend.upper()}: Top View - Swing Development', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 22)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        swing_cm = abs(y[-1] - y[0]) * 100
        ax1.text(0.95, 0.95, f'Total Swing: {swing_cm:.2f} cm', 
                transform=ax1.transAxes, ha='right', va='top',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Side view (flight)
        ax2 = axes[idx, 1]
        ax2.plot(x, z, 'b-', linewidth=2.5)
        ax2.scatter(x[0], z[0], c='green', s=150, marker='o', zorder=5, label='Release')
        ax2.scatter(x[-1], z[-1], c='red', s=150, marker='X', zorder=5, label='Landing')
        ax2.axhline(0, color='brown', linewidth=4, label='Pitch')
        ax2.fill_between(x, 0, z, alpha=0.2, color='blue')
        ax2.set_xlabel('Distance (m)', fontsize=12)
        ax2.set_ylabel('Height (m)', fontsize=12)
        ax2.set_title(f'{backend.upper()}: Side View - Ball Flight', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 22)
        ax2.set_ylim(0, 3)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Reset tesseracts for next backend
        from helper.docker import _INTEGRATOR, _SWING, _OPTIMIZER
        import helper.docker as docker_module
        docker_module._INTEGRATOR = None
        docker_module._SWING = None
        docker_module._OPTIMIZER = None

    plt.suptitle('Cricket Ball Trajectory Comparison\nVelocity: 126 km/h | Roughness: 0.8 | Seam: 30°', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150, bbox_inches='tight')
    print("Saved comparison_plot.png")
    
finally:
    cleanup_containers()
