"""
🏏 SUPER FANCY DeepSwingr Analysis 🏏
Comprehensive cricket ball swing visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from helper.docker import get_tesseracts, cleanup_containers, PHYSICS_BACKENDS
import warnings
warnings.filterwarnings('ignore')

print("🏏 " + "="*60 + " 🏏")
print("   SUPER FANCY DeepSwingr Analysis")
print("   Comprehensive Cricket Ball Swing Visualization")
print("🏏 " + "="*60 + " 🏏\n")

# Use jaxphysics for more realistic results
backend = "jaxphysics"
url = PHYSICS_BACKENDS[backend]["url"]

try:
    integrator, swing, optimizer = get_tesseracts(backend_name=backend)
    
    # ========================================================================
    # 1. FAMOUS DELIVERIES COMPARISON
    # ========================================================================
    print("\n📊 [1/4] Simulating Famous Delivery Types...")
    
    deliveries = {
        "Wasim Akram Inswinger": {"velocity": 40, "seam": -25, "roughness": 0.7, "color": "red"},
        "Glenn McGrath Outswinger": {"velocity": 38, "seam": 20, "roughness": 0.6, "color": "blue"},
        "Dale Steyn Express": {"velocity": 45, "seam": 15, "roughness": 0.8, "color": "green"},
        "James Anderson Wobble": {"velocity": 36, "seam": 35, "roughness": 0.9, "color": "purple"},
    }
    
    delivery_results = {}
    for name, params in deliveries.items():
        print(f"   Simulating {name}...")
        res = integrator.apply({
            "initial_velocity": float(params["velocity"]),
            "release_angle": 5.0,
            "roughness": params["roughness"],
            "seam_angle": float(params["seam"]),
            "physics_url": url
        })
        delivery_results[name] = {
            "x": np.array(res["x_positions"]),
            "y": np.array(res["y_positions"]),
            "z": np.array(res["z_positions"]),
            "params": params
        }
    
    # ========================================================================
    # 2. SEAM ANGLE SWEEP
    # ========================================================================
    print("\n📊 [2/4] Computing Seam Angle Sensitivity...")
    
    seam_angles = np.linspace(-60, 60, 13)
    angle_swings = []
    for angle in seam_angles:
        res = swing.apply({
            "initial_velocity": 38.0,
            "release_angle": 5.0,
            "roughness": 0.8,
            "seam_angle": float(angle),
            "physics_url": url
        })
        angle_swings.append(res["final_deviation"])
        print(f"   Seam {angle:+6.1f}° → Swing: {res['final_deviation']:+.2f} cm")
    
    # ========================================================================
    # 3. VELOCITY VS SWING ANALYSIS
    # ========================================================================
    print("\n📊 [3/4] Computing Velocity vs Swing...")
    
    velocities = np.linspace(30, 45, 8)
    velocity_swings = []
    for v in velocities:
        res = swing.apply({
            "initial_velocity": float(v),
            "release_angle": 5.0,
            "roughness": 0.8,
            "seam_angle": 25.0,
            "physics_url": url
        })
        velocity_swings.append(res["final_deviation"])
        print(f"   {v*3.6:.0f} km/h → Swing: {res['final_deviation']:.2f} cm")
    
    # ========================================================================
    # 4. OPTIMIZATION SURFACE (Seam vs Roughness)
    # ========================================================================
    print("\n📊 [4/4] Computing Optimization Surface...")
    
    seam_grid = np.linspace(-45, 45, 10)
    rough_grid = np.linspace(0.3, 1.0, 8)
    swing_surface = np.zeros((len(rough_grid), len(seam_grid)))
    
    for i, rough in enumerate(rough_grid):
        for j, seam in enumerate(seam_grid):
            res = swing.apply({
                "initial_velocity": 38.0,
                "release_angle": 5.0,
                "roughness": float(rough),
                "seam_angle": float(seam),
                "physics_url": url
            })
            swing_surface[i, j] = abs(res["final_deviation"])
        print(f"   Roughness {rough:.2f} complete...")
    
    # ========================================================================
    # CREATE MEGA VISUALIZATION
    # ========================================================================
    print("\n🎨 Creating Super Fancy Visualization...")
    
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0a0a0a')
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Style settings
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    
    # ---- Panel 1: Famous Deliveries (Top View) ----
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.set_facecolor('#1a1a2e')
    
    # Cricket pitch
    pitch_rect = Rectangle((0, -1.22), 20.12, 2.44, fill=True, 
                           facecolor='#2d5016', edgecolor='white', linewidth=2)
    ax1.add_patch(pitch_rect)
    
    # Crease lines
    ax1.axvline(x=1.22, color='white', linewidth=2, linestyle='-')
    ax1.axvline(x=18.9, color='white', linewidth=2, linestyle='-')
    
    # Stumps
    for stump_x in [0.3, 20.12-0.3]:
        for stump_y in [-0.1, 0, 0.1]:
            ax1.plot(stump_x, stump_y, 'o', color='#d4af37', markersize=8)
    
    for name, data in delivery_results.items():
        x, y = data["x"], data["y"]
        color = data["params"]["color"]
        
        # Create gradient line
        points = np.array([x, y*100]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.get_cmap('plasma'), linewidth=3)
        lc.set_array(np.linspace(0, 1, len(x)))
        ax1.add_collection(lc)
        
        ax1.plot(x, y*100, color=color, linewidth=3, alpha=0.8, label=name)
        ax1.scatter(x[-1], y[-1]*100, color=color, s=200, marker='X', edgecolors='white', linewidth=2, zorder=10)
    
    ax1.set_xlim(-2, 24)
    ax1.set_ylim(-30, 30)
    ax1.set_xlabel('Distance (m)', fontsize=12, color='white')
    ax1.set_ylabel('Lateral Deviation (cm)', fontsize=12, color='white')
    ax1.set_title('🏏 Famous Delivery Types Comparison', fontsize=16, fontweight='bold', color='#00ff88', pad=10)
    ax1.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', fontsize=10)
    ax1.grid(True, alpha=0.2, color='white')
    
    # ---- Panel 2: Seam Angle Sensitivity ----
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#1a1a2e')
    
    colors = cm.coolwarm(np.linspace(0, 1, len(seam_angles)))
    bars = ax2.bar(seam_angles, angle_swings, width=8, color=colors, edgecolor='white', linewidth=1)
    ax2.axhline(0, color='white', linewidth=1, linestyle='--')
    ax2.set_xlabel('Seam Angle (°)', fontsize=12, color='white')
    ax2.set_ylabel('Swing (cm)', fontsize=12, color='white')
    ax2.set_title('🎯 Seam Angle Sensitivity', fontsize=14, fontweight='bold', color='#ff6b6b', pad=10)
    ax2.grid(True, alpha=0.2, color='white', axis='y')
    
    # ---- Panel 3: Velocity vs Swing ----
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#1a1a2e')
    
    vel_kmh = velocities * 3.6
    ax3.fill_between(vel_kmh, 0, velocity_swings, alpha=0.4, color='#00d4ff')
    ax3.plot(vel_kmh, velocity_swings, 'o-', color='#00d4ff', linewidth=3, markersize=10, markeredgecolor='white')
    ax3.set_xlabel('Bowling Speed (km/h)', fontsize=12, color='white')
    ax3.set_ylabel('Swing (cm)', fontsize=12, color='white')
    ax3.set_title('⚡ Speed vs Swing', fontsize=14, fontweight='bold', color='#00d4ff', pad=10)
    ax3.grid(True, alpha=0.2, color='white')
    
    # ---- Panel 4: 3D Trajectory ----
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    ax4.set_facecolor('#1a1a2e')
    
    # Plot best delivery trajectory
    best_name = "Dale Steyn Express"
    data = delivery_results[best_name]
    x, y, z = data["x"], data["y"], data["z"]
    
    # 3D trajectory with color gradient
    for i in range(len(x)-1):
        ax4.plot3D(x[i:i+2], y[i:i+2]*100, z[i:i+2], 
                  color=cm.hot(i/len(x)), linewidth=3)
    
    ax4.scatter3D([x[0]], [y[0]*100], [z[0]], color='green', s=200, marker='o', label='Release')
    ax4.scatter3D([x[-1]], [y[-1]*100], [z[-1]], color='red', s=200, marker='X', label='Landing')
    
    # Pitch surface
    pitch_x = np.linspace(0, 20.12, 10)
    pitch_y = np.linspace(-20, 20, 10)
    pitch_X, pitch_Y = np.meshgrid(pitch_x, pitch_y)
    pitch_Z = np.zeros_like(pitch_X)
    ax4.plot_surface(pitch_X, pitch_Y, pitch_Z, alpha=0.3, color='green')
    
    ax4.set_xlabel('Distance (m)', color='white')
    ax4.set_ylabel('Lateral (cm)', color='white')
    ax4.set_zlabel('Height (m)', color='white')
    ax4.set_title(f'🚀 3D Trajectory: {best_name}', fontsize=14, fontweight='bold', color='#ffd700', pad=10)
    ax4.view_init(elev=20, azim=-60)
    
    # ---- Panel 5: Optimization Surface ----
    ax5 = fig.add_subplot(gs[1, 2], projection='3d')
    ax5.set_facecolor('#1a1a2e')
    
    SEAM, ROUGH = np.meshgrid(seam_grid, rough_grid)
    surf = ax5.plot_surface(SEAM, ROUGH, swing_surface, cmap='magma', 
                           edgecolor='white', linewidth=0.2, alpha=0.9)
    
    # Find and mark optimal point
    max_idx = np.unravel_index(swing_surface.argmax(), swing_surface.shape)
    opt_seam = seam_grid[max_idx[1]]
    opt_rough = rough_grid[max_idx[0]]
    opt_swing = swing_surface[max_idx]
    ax5.scatter3D([opt_seam], [opt_rough], [opt_swing], color='cyan', s=300, marker='*', edgecolors='white')
    
    ax5.set_xlabel('Seam Angle (°)', color='white')
    ax5.set_ylabel('Roughness', color='white')
    ax5.set_zlabel('|Swing| (cm)', color='white')
    ax5.set_title('🔥 Optimization Surface', fontsize=14, fontweight='bold', color='#ff4500', pad=10)
    ax5.view_init(elev=25, azim=45)
    
    # ---- Panel 6: Physics Explanation ----
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.set_facecolor('#1a1a2e')
    ax6.axis('off')
    
    explanation = """
    🎓 THE PHYSICS OF CRICKET BALL SWING
    
    ▸ CONVENTIONAL SWING (Low Speed)
      • Ball swings toward the shiny side
      • Seam trips boundary layer on rough side
      • Pressure differential creates lateral force
    
    ▸ REVERSE SWING (High Speed)  
      • Ball swings toward the rough side
      • Turbulent boundary layer on both sides
      • Delayed separation on rough side
    
    ▸ KEY PARAMETERS
      • Seam angle: 15-25° optimal
      • Speed: 130-145 km/h sweet spot
      • Roughness: Asymmetric wear crucial
    """
    ax6.text(0.05, 0.95, explanation, transform=ax6.transAxes, fontsize=11,
            color='white', family='monospace', va='top',
            bbox=dict(boxstyle='round', facecolor='#2a2a4a', edgecolor='#00ff88', linewidth=2))
    
    # ---- Panel 7: Delivery Stats ----
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.set_facecolor('#1a1a2e')
    ax7.axis('off')
    
    stats_text = "📊 DELIVERY STATISTICS\n" + "─"*40 + "\n\n"
    for name, data in delivery_results.items():
        swing_cm = abs(data["y"][-1] - data["y"][0]) * 100
        speed_kmh = data["params"]["velocity"] * 3.6
        stats_text += f"  {name}\n"
        stats_text += f"    Speed: {speed_kmh:.0f} km/h | Swing: {swing_cm:.1f} cm\n\n"
    
    stats_text += f"\n  🏆 OPTIMAL PARAMETERS (Max Swing)\n"
    stats_text += f"    Seam: {opt_seam:.1f}° | Roughness: {opt_rough:.2f}\n"
    stats_text += f"    Maximum Swing: {opt_swing:.1f} cm"
    
    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=11,
            color='white', family='monospace', va='top',
            bbox=dict(boxstyle='round', facecolor='#2a2a4a', edgecolor='#ffd700', linewidth=2))
    
    # ---- Panel 8: Ball Diagram ----
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.set_facecolor('#1a1a2e')
    ax8.set_aspect('equal')
    
    # Draw cricket ball
    ball = Circle((0.5, 0.5), 0.35, facecolor='#8B0000', edgecolor='white', linewidth=3)
    ax8.add_patch(ball)
    
    # Seam line
    seam_angle_rad = np.deg2rad(25)
    ax8.plot([0.5 - 0.3*np.cos(seam_angle_rad), 0.5 + 0.3*np.cos(seam_angle_rad)],
            [0.5 - 0.3*np.sin(seam_angle_rad), 0.5 + 0.3*np.sin(seam_angle_rad)],
            color='white', linewidth=4, solid_capstyle='round')
    
    # Stitching
    for offset in [-0.02, 0.02]:
        ax8.plot([0.5 - 0.28*np.cos(seam_angle_rad) + offset*np.sin(seam_angle_rad), 
                 0.5 + 0.28*np.cos(seam_angle_rad) + offset*np.sin(seam_angle_rad)],
                [0.5 - 0.28*np.sin(seam_angle_rad) - offset*np.cos(seam_angle_rad), 
                 0.5 + 0.28*np.sin(seam_angle_rad) - offset*np.cos(seam_angle_rad)],
                color='#FFD700', linewidth=2, linestyle='--')
    
    # Airflow arrows
    ax8.annotate('', xy=(0.15, 0.5), xytext=(-0.1, 0.5),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=3))
    ax8.annotate('', xy=(0.15, 0.7), xytext=(-0.1, 0.7),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=2))
    ax8.annotate('', xy=(0.15, 0.3), xytext=(-0.1, 0.3),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=2))
    
    # Swing force arrow
    ax8.annotate('', xy=(0.5, 0.95), xytext=(0.5, 0.85),
                arrowprops=dict(arrowstyle='->', color='#00ff88', lw=4))
    ax8.text(0.5, 1.0, 'SWING', ha='center', fontsize=12, color='#00ff88', fontweight='bold')
    ax8.text(0.5, -0.05, 'AIRFLOW →', ha='center', fontsize=10, color='cyan')
    
    ax8.set_xlim(-0.2, 1.2)
    ax8.set_ylim(-0.15, 1.15)
    ax8.axis('off')
    ax8.set_title('🔴 Cricket Ball Aerodynamics', fontsize=14, fontweight='bold', color='#ff6b6b', pad=10)
    
    # ---- Main Title ----
    fig.suptitle('🏏 DeepSwingr: Advanced Cricket Ball Swing Analysis 🏏\n'
                'Powered by JAX-CFD Differentiable Physics Engine',
                fontsize=20, fontweight='bold', color='white', y=0.98)
    
    plt.savefig('super_fancy_analysis.png', dpi=150, bbox_inches='tight', 
                facecolor='#0a0a0a', edgecolor='none')
    print("\n✨ Saved: super_fancy_analysis.png")
    
    # Also save as interactive HTML
    print("\n📊 Creating interactive Plotly dashboard...")
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig_plotly = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "surface"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=("3D Ball Trajectory", "Swing Optimization Surface",
                       "Famous Deliveries", "Speed vs Swing")
    )
    
    # 3D Trajectory
    data = delivery_results["Dale Steyn Express"]
    fig_plotly.add_trace(go.Scatter3d(
        x=data["x"], y=data["y"]*100, z=data["z"],
        mode='lines', line=dict(color='red', width=6),
        name='Dale Steyn Express'
    ), row=1, col=1)
    
    # Optimization surface
    fig_plotly.add_trace(go.Surface(
        x=seam_grid, y=rough_grid, z=swing_surface,
        colorscale='Magma', showscale=True, name='Swing Surface'
    ), row=1, col=2)
    
    # Famous deliveries
    for name, data in delivery_results.items():
        fig_plotly.add_trace(go.Scatter(
            x=data["x"], y=data["y"]*100,
            mode='lines', name=name, line=dict(width=3)
        ), row=2, col=1)
    
    # Speed vs swing
    fig_plotly.add_trace(go.Scatter(
        x=velocities*3.6, y=velocity_swings,
        mode='lines+markers', name='Speed vs Swing',
        line=dict(color='cyan', width=3),
        marker=dict(size=10)
    ), row=2, col=2)
    
    fig_plotly.update_layout(
        title="🏏 DeepSwingr Interactive Dashboard",
        template="plotly_dark",
        height=900,
        showlegend=True
    )
    
    fig_plotly.write_html("super_fancy_dashboard.html")
    print("✨ Saved: super_fancy_dashboard.html")
    
finally:
    cleanup_containers()

print("\n" + "🏏 "*20)
print("   ANALYSIS COMPLETE!")
print("🏏 "*20)
