"""
🏏 DeepSwingr - Pure JAX Cricket Ball Simulator
No Docker • No Tesseract • ~100x faster!
"""

import numpy as np
from simulator import simulate_trajectory, compute_swing, optimize_seam_angle
from plotter.plt import plot_trajectory_3d, plot_optimization_results

BACKENDS = ["jaxphysics", "simplephysics", "analytical"]


def show_trajectory(v=35.0, a=5.0, r=0.8, s=30.0, backend="jaxphysics"):
    print(f"\n🎾 Simulating: {v*3.6:.0f} km/h, seam={s:.0f}°, roughness={r:.2f}")
    times, x, y, z, vel = simulate_trajectory(v, a, r, s, backend)
    print(f"   ✓ Swing: {(y[-1]-y[0])*100:.2f} cm")
    plot_trajectory_3d(np.array(times), np.array(x), np.array(y), np.array(z), 
                       np.array(vel), v, r, s, use_plotly=True).show()


def plot_optimal(r=0.8, a=5.0, swing_type="out", speed_range=[30, 40], n=5, backend="jaxphysics"):
    print(f"\n⚡ Optimizing {swing_type}swing...")
    speeds = np.linspace(speed_range[0], speed_range[1], n)
    results = [optimize_seam_angle(float(v), a, r, swing_type, backend) for v in speeds]
    angles, devs = zip(*results)
    for v, dev, ang in zip(speeds, devs, angles):
        print(f"  {v*3.6:.0f} km/h: {dev:.1f}cm at {ang:.0f}°")
    plot_optimization_results(speeds, np.array(devs), np.array(angles), swing_type, r)


def main():
    backend = "jaxphysics"
    print("\n" + "="*50)
    print("  🏏 DeepSwingr - Pure JAX (No Docker!)")
    print("="*50)
    
    while True:
        print(f"\n[{backend}] 1.Trajectory 2.Optimize 3.Backend 4.Exit")
        c = input("Choice: ").strip()
        
        if c == "4": break
        elif c == "3":
            for i, b in enumerate(BACKENDS): print(f"  {i+1}. {b}")
            try: backend = BACKENDS[int(input("Choice: "))-1]
            except: pass
        elif c == "1":
            show_trajectory(
                float(input("Velocity m/s [35]: ") or 35),
                float(input("Angle deg [5]: ") or 5),
                float(input("Roughness [0.8]: ") or 0.8),
                float(input("Seam deg [30]: ") or 30),
                backend
            )
        elif c == "2":
            plot_optimal(
                float(input("Roughness [0.8]: ") or 0.8),
                swing_type=input("Type (in/out) [out]: ") or "out",
                backend=backend
            )

if __name__ == "__main__":
    main()

