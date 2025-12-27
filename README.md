# 🏏 DeepSwingr - Pure JAX

**Cricket Ball Swing Simulation - No Docker, No Tesseract, No Bullshit!**

This is a clean rewrite of [DeepSwingr](https://github.com/gpavanb1/DeepSwingr) that removes all containerization overhead while keeping the same physics accuracy.

## Why This Version?

| Original (Tesseract) | This Version (Pure JAX) |
|---------------------|------------------------|
| 5 Docker containers | 0 containers |
| HTTP calls per ODE step | Direct function calls |
| ~500ms per simulation | ~5ms per simulation |
| Complex setup | `pip install -r requirements.txt` |

**~100x faster** because we removed the HTTP/serialization overhead from the ODE inner loop.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python demo_run.py

# Interactive CLI
python main.py

# Test physics models
python physics.py
python simulator.py
```

## Project Structure

```
DeepSwingr-clean/
├── physics.py          # Neural network physics models
├── simulator.py        # Trajectory simulation (Diffrax)
├── main.py            # Interactive CLI
├── demo_run.py        # Demo script
├── weights/           # Pre-trained model weights
│   ├── jaxphysics_weights.msgpack     # 138KB - JAX-CFD trained
│   └── simplephysics_weights.msgpack  # 36KB - Simpler model
├── plotter/           # Visualization (unchanged)
└── deploy/            # Remote deployment options
    ├── modal_app.py   # Serverless (Modal.com)
    ├── Dockerfile     # Container deployment
    └── README.md      # Deployment guide
```

## Physics Backends

1. **jaxphysics** (default): Larger NN trained on JAX-CFD simulations
2. **simplephysics**: Smaller NN, faster but less accurate  
3. **analytical**: Empirical formulas (no weights needed)

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

If you need remote compute (you probably don't - inference is very light):

```bash
# Modal (simplest)
pip install modal
modal run deploy/modal_app.py

# Docker
docker build -t deepswingr -f deploy/Dockerfile .
docker run -p 8000:8000 deepswingr
curl "http://localhost:8000/simulate?velocity=35&seam_angle=30"
```

See `deploy/README.md` for more options (AWS, GCP, Ray).

## The Key Insight

The original Tesseract version containerized everything for "dependency isolation", but:

1. **Training** (heavy CFD) → One-time, already done, weights saved
2. **Inference** (neural network) → Just matrix multiplies, runs anywhere

The trained weights are only **138KB**. You don't need 5 Docker containers to multiply matrices.

## License

Apache 2.0 (same as original)

