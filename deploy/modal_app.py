"""
🚀 DeepSwingr on Modal - Serverless Remote Compute
The SIMPLEST way to run heavy physics remotely.

Setup (one time):
    pip install modal
    modal token new

Run:
    modal run modal_app.py

Deploy as API:
    modal deploy modal_app.py
    # Then call: https://your-username--deepswingr-api.modal.run/simulate?velocity=35&seam_angle=30
"""

import modal

# Define the Modal app and container image
app = modal.App("deepswingr")

# Container image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "jax[cpu]>=0.4.20",
    "flax>=0.8.0", 
    "diffrax>=0.5.0",
    "numpy>=1.24.0",
)

# Mount local weights
weights_mount = modal.Mount.from_local_dir("../weights", remote_path="/app/weights")


@app.function(image=image, mounts=[weights_mount])
def simulate_remote(velocity: float, release_angle: float, roughness: float, seam_angle: float):
    """Run simulation on Modal's cloud infrastructure."""
    import sys
    sys.path.insert(0, "/app")
    
    # Inline the physics and simulator to avoid import issues
    import jax
    import jax.numpy as jnp
    from jax import jit
    import flax.linen as nn
    import flax.serialization
    import diffrax as dfx
    from pathlib import Path
    
    # Physics model
    class JaxPhysicsNetwork(nn.Module):
        hidden_dims = (64, 128, 128, 64)
        @nn.compact
        def __call__(self, x):
            roughness, angle, re = x[0], x[1], x[2]
            angle_rad = jnp.deg2rad(angle)
            x_norm = jnp.array([roughness, jnp.sin(angle_rad), jnp.cos(angle_rad), 
                               jnp.log10(re)/6.0, roughness * jnp.sin(angle_rad)])
            for dim in self.hidden_dims:
                x_norm = nn.gelu(nn.LayerNorm()(nn.Dense(dim)(x_norm)))
            return nn.Dense(3)(x_norm)
    
    # Load weights
    model = JaxPhysicsNetwork()
    dummy_params = model.init(jax.random.PRNGKey(0), jnp.ones(3))
    with open("/app/weights/jaxphysics_weights.msgpack", "rb") as f:
        params = flax.serialization.from_bytes(dummy_params, f.read())
    
    forward = jit(lambda p, x: model.apply(p, x))
    
    # Simulate
    MASS, DIAMETER, RHO_AIR, MU, G, PITCH = 0.156, 0.07, 1.225, 1.5e-5, 9.81, 20.12
    
    @jit
    def dynamics(t, y, args):
        x, yp, z, vx, vy, vz = y
        v_mag = jnp.sqrt(vx**2 + vy**2 + vz**2)
        Re = jnp.clip(RHO_AIR * v_mag * DIAMETER / MU, 1e5, 1e6)
        forces = forward(params, jnp.array([roughness, seam_angle, Re]))
        vs = v_mag + 1e-6
        Fx, Fy, Fz = -forces[0]*vx/vs, -forces[0]*vy/vs + forces[2], -forces[0]*vz/vs + forces[1] - MASS*G
        active = (z > 0) & (x < PITCH)
        return jnp.where(active, jnp.array([vx, vy, vz, Fx/MASS, Fy/MASS, Fz/MASS]), jnp.zeros(6))
    
    theta = jnp.deg2rad(release_angle)
    y0 = jnp.array([0., 0., 2., velocity*jnp.cos(theta), 0., velocity*jnp.sin(theta)])
    
    sol = dfx.diffeqsolve(dfx.ODETerm(dynamics), dfx.Heun(), t0=0., t1=1.2, dt0=0.002, y0=y0,
                          saveat=dfx.SaveAt(ts=jnp.linspace(0, 1.2, 500)),
                          stepsize_controller=dfx.ConstantStepSize(), max_steps=5000)
    
    swing_cm = float((sol.ys[-1, 1] - sol.ys[0, 1]) * 100)
    return {
        "swing_cm": swing_cm,
        "final_x": float(sol.ys[-1, 0]),
        "final_z": float(sol.ys[-1, 2]),
        "times": sol.ts.tolist(),
        "x": sol.ys[:, 0].tolist(),
        "y": sol.ys[:, 1].tolist(),
        "z": sol.ys[:, 2].tolist(),
    }


@app.function(image=image, mounts=[weights_mount])
def optimize_remote(velocity: float, roughness: float, swing_type: str = "out"):
    """Find optimal seam angle remotely."""
    import numpy as np
    
    best_angle, best_swing = 0.0, 0.0
    for angle in np.linspace(-90, 90, 37):
        result = simulate_remote.local(velocity, 5.0, roughness, float(angle))
        swing = result["swing_cm"]
        if (swing_type == "out" and swing > best_swing) or (swing_type == "in" and swing < -best_swing):
            best_swing = abs(swing) if swing_type == "in" else swing
            best_angle = angle
    
    return {"optimal_angle": best_angle, "max_swing_cm": best_swing}


# FastAPI endpoint for web access
@app.function(image=image, mounts=[weights_mount])
@modal.web_endpoint(method="GET")
def api(velocity: float = 35.0, release_angle: float = 5.0, roughness: float = 0.8, seam_angle: float = 30.0):
    """HTTP API endpoint: /simulate?velocity=35&seam_angle=30"""
    return simulate_remote.local(velocity, release_angle, roughness, seam_angle)


@app.local_entrypoint()
def main():
    """Run from command line: modal run modal_app.py"""
    print("🏏 Running DeepSwingr on Modal...")
    result = simulate_remote.remote(35.0, 5.0, 0.8, 30.0)
    print(f"✓ Swing: {result['swing_cm']:.2f} cm")
    
    print("\n⚡ Finding optimal seam angle...")
    opt = optimize_remote.remote(35.0, 0.8, "out")
    print(f"✓ Optimal: {opt['optimal_angle']:.0f}° → {opt['max_swing_cm']:.1f} cm")

