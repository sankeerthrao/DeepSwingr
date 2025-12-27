# 🏏 DeepSwingr - Pure JAX Edition

A differentiable cricket ball swing simulation using JAX, Flax, and Diffrax.

Based on [gpavanb1/DeepSwingr](https://github.com/gpavanb1/DeepSwingr) - refactored to run as pure Python/JAX without containerization.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python demo_run.py

# Interactive CLI
python main.py
```

## Project Structure

```
DeepSwingr/
├── physics.py          # Neural network physics models
├── simulator.py        # Trajectory simulation (Diffrax)
├── main.py             # Interactive CLI
├── demo_run.py         # Demo script
├── weights/            # Pre-trained model weights
│   ├── jaxphysics_weights.msgpack     # Trained on JAX-CFD
│   └── simplephysics_weights.msgpack  # Simpler model
├── plotter/            # Visualization
└── deploy/             # Remote deployment options
    ├── modal_app.py    # Serverless (Modal.com)
    ├── Dockerfile      # Container deployment
    └── README.md       # Deployment guide
```

## Physics Backends

1. **jaxphysics** (default): Larger neural network trained on JAX-CFD simulations
2. **simplephysics**: Smaller neural network, faster inference
3. **analytical**: Empirical formulas based on Mehta (1985)

## Usage

```python
from simulator import simulate_trajectory, compute_swing, optimize_seam_angle

# Single simulation
times, x, y, z, velocities = simulate_trajectory(
    initial_velocity=35.0,  # m/s
    release_angle=5.0,      # degrees
    roughness=0.8,          # 0-1
    seam_angle=30.0,        # degrees
    backend="jaxphysics"
)
print(f"Swing: {(y[-1]-y[0])*100:.2f} cm")

# Compute swing directly
swing_cm = compute_swing(35.0, 5.0, 0.8, 30.0, "jaxphysics")

# Find optimal seam angle
optimal_angle, max_swing = optimize_seam_angle(35.0, 5.0, 0.8, "out", "jaxphysics")
```

## Remote Deployment

For remote compute options, see `deploy/README.md`:

```bash
# Modal (serverless)
pip install modal
modal run deploy/modal_app.py

# Docker
docker build -t deepswingr -f deploy/Dockerfile .
docker run -p 8000:8000 deepswingr
```

## Credits

- Original implementation: [gpavanb1/DeepSwingr](https://github.com/gpavanb1/DeepSwingr)
- Physics models trained using [JAX-CFD](https://github.com/google/jax-cfd)
- Empirical formulas based on Mehta (1985), Barton (1982)

## License

Apache 2.0
