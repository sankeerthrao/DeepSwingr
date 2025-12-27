# DeepSwingr - Simple Docker for Remote Deployment
# Build: docker build -t deepswingr .
# Run:   docker run -p 8000:8000 deepswingr

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# Copy code and weights
COPY physics.py simulator.py ./
COPY weights/ ./weights/

# Create simple FastAPI server
RUN cat > server.py << 'EOF'
from fastapi import FastAPI
from simulator import simulate_trajectory, compute_swing, optimize_seam_angle
import numpy as np

app = FastAPI(title="DeepSwingr API")

@app.get("/simulate")
def simulate(velocity: float = 35.0, release_angle: float = 5.0, 
             roughness: float = 0.8, seam_angle: float = 30.0, backend: str = "jaxphysics"):
    times, x, y, z, vel = simulate_trajectory(velocity, release_angle, roughness, seam_angle, backend)
    return {
        "swing_cm": float((y[-1] - y[0]) * 100),
        "times": [float(t) for t in times],
        "x": [float(v) for v in x],
        "y": [float(v) for v in y],
        "z": [float(v) for v in z],
    }

@app.get("/optimize")
def optimize(velocity: float = 35.0, roughness: float = 0.8, swing_type: str = "out", backend: str = "jaxphysics"):
    angle, swing = optimize_seam_angle(velocity, 5.0, roughness, swing_type, backend)
    return {"optimal_angle": angle, "max_swing_cm": swing}

@app.get("/health")
def health():
    return {"status": "ok"}
EOF

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

