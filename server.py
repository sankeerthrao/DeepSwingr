"""
DeepSwingr FastAPI Server
Run with: uvicorn server:app --reload --port 8000
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

from simulator import simulate_trajectory, compute_swing, optimize_seam_angle
from physics import PhysicsModel, analytical_physics

app = FastAPI(
    title="DeepSwingr API",
    description="Cricket ball swing simulation using differentiable physics",
    version="1.0.0",
)

# Allow CORS for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preload models on startup
@app.on_event("startup")
async def startup():
    print("Loading physics models...")
    PhysicsModel.load("jaxphysics")
    PhysicsModel.load("simplephysics")
    print("✓ Models loaded")


# ============================================================================
# API MODELS
# ============================================================================

class SimulationRequest(BaseModel):
    initial_velocity: float = Field(35.0, description="Ball speed in m/s", ge=20, le=50)
    release_angle: float = Field(5.0, description="Release angle in degrees", ge=0, le=15)
    roughness: float = Field(0.8, description="Surface roughness 0-1", ge=0, le=1)
    seam_angle: float = Field(30.0, description="Seam angle in degrees", ge=-90, le=90)
    backend: str = Field("jaxphysics", description="Physics backend")

class SimulationResponse(BaseModel):
    swing_cm: float
    pitch_distance_m: float
    flight_time_s: float
    final_speed_kmh: float
    times: List[float]
    x_positions: List[float]
    y_positions: List[float]
    z_positions: List[float]

class OptimizationRequest(BaseModel):
    initial_velocity: float = Field(35.0, ge=20, le=50)
    release_angle: float = Field(5.0, ge=0, le=15)
    roughness: float = Field(0.8, ge=0, le=1)
    swing_type: str = Field("out", description="'out' or 'in'")
    backend: str = Field("jaxphysics")

class OptimizationResponse(BaseModel):
    optimal_seam_angle: float
    maximum_swing_cm: float

class ForcesRequest(BaseModel):
    notch_angle: float = Field(30.0, ge=-90, le=90)
    reynolds_number: float = Field(1.5e5, ge=1e4, le=1e7)
    roughness: float = Field(0.8, ge=0, le=1)
    backend: str = Field("jaxphysics")

class ForcesResponse(BaseModel):
    drag: float
    lift: float
    side: float


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "DeepSwingr API",
        "version": "1.0.0",
        "endpoints": ["/simulate", "/optimize", "/forces", "/health"],
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/simulate", response_model=SimulationResponse)
async def simulate(req: SimulationRequest):
    """Simulate ball trajectory and return full path."""
    times, x, y, z, vel = simulate_trajectory(
        req.initial_velocity, req.release_angle, 
        req.roughness, req.seam_angle, req.backend
    )
    
    times, x, y, z, vel = np.array(times), np.array(x), np.array(y), np.array(z), np.array(vel)
    
    # Find valid trajectory (before hitting ground)
    valid_idx = np.where((z > 0) & (x < 20.12))[0]
    last_idx = valid_idx[-1] if len(valid_idx) > 0 else len(x) - 1
    
    return SimulationResponse(
        swing_cm=float((y[last_idx] - y[0]) * 100),
        pitch_distance_m=float(x[last_idx]),
        flight_time_s=float(times[last_idx]),
        final_speed_kmh=float(np.linalg.norm(vel[last_idx]) * 3.6),
        times=times.tolist(),
        x_positions=x.tolist(),
        y_positions=y.tolist(),
        z_positions=z.tolist(),
    )

@app.get("/simulate")
async def simulate_get(
    velocity: float = Query(35.0, ge=20, le=50),
    release_angle: float = Query(5.0, ge=0, le=15),
    roughness: float = Query(0.8, ge=0, le=1),
    seam_angle: float = Query(30.0, ge=-90, le=90),
    backend: str = Query("jaxphysics"),
):
    """GET endpoint for easy browser testing."""
    req = SimulationRequest(
        initial_velocity=velocity, release_angle=release_angle,
        roughness=roughness, seam_angle=seam_angle, backend=backend
    )
    return await simulate(req)

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize(req: OptimizationRequest):
    """Find optimal seam angle for maximum swing."""
    angle, swing = optimize_seam_angle(
        req.initial_velocity, req.release_angle,
        req.roughness, req.swing_type, req.backend
    )
    return OptimizationResponse(
        optimal_seam_angle=angle,
        maximum_swing_cm=swing
    )

@app.post("/forces", response_model=ForcesResponse)
async def forces(req: ForcesRequest):
    """Get raw force values from physics model."""
    if req.backend == "analytical":
        f = analytical_physics(req.notch_angle, req.reynolds_number, req.roughness)
    else:
        model = PhysicsModel.load(req.backend)
        f = model(req.notch_angle, req.reynolds_number, req.roughness)
    
    return ForcesResponse(drag=float(f[0]), lift=float(f[1]), side=float(f[2]))

@app.get("/backends")
async def list_backends():
    """List available physics backends."""
    return {
        "backends": ["jaxphysics", "simplephysics", "analytical"],
        "default": "jaxphysics",
        "descriptions": {
            "jaxphysics": "Neural network trained on JAX-CFD (64-128-128-64)",
            "simplephysics": "Neural network trained on empirical formulas (32-64-64-32)",
            "analytical": "Direct empirical formulas from Mehta (1985)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

