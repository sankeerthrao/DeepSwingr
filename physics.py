"""
Cricket Ball Aerodynamics - Pure JAX Physics Models
No Docker, no HTTP, no Tesseract - just JAX!

Two models available:
- simplephysics: Smaller network (32, 64, 64, 32), empirical CFD
- jaxphysics: Larger network (64, 128, 128, 64), trained on JAX-CFD
"""

import jax
import jax.numpy as jnp
from jax import jit
import flax.linen as nn
import flax.serialization
from pathlib import Path
from typing import Tuple, Optional

# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class SimplePhysicsNetwork(nn.Module):
    """Smaller network for simplephysics backend."""
    hidden_dims: Tuple[int, ...] = (32, 64, 64, 32)

    @nn.compact
    def __call__(self, x):
        roughness, angle, re = x[0], x[1], x[2]
        re_normalized = jnp.log10(re) / 6.0
        angle_rad = jnp.deg2rad(angle)
        x_norm = jnp.array([roughness, jnp.sin(angle_rad), jnp.cos(angle_rad), re_normalized])

        for dim in self.hidden_dims:
            x_norm = nn.gelu(nn.LayerNorm()(nn.Dense(dim)(x_norm)))
        return nn.Dense(3)(x_norm)


class JaxPhysicsNetwork(nn.Module):
    """Larger network for jaxphysics backend (trained on JAX-CFD)."""
    hidden_dims: Tuple[int, ...] = (64, 128, 128, 64)

    @nn.compact
    def __call__(self, x):
        roughness, angle, re = x[0], x[1], x[2]
        re_normalized = jnp.log10(re) / 6.0
        angle_rad = jnp.deg2rad(angle)
        x_norm = jnp.array([
            roughness, jnp.sin(angle_rad), jnp.cos(angle_rad), 
            re_normalized, roughness * jnp.sin(angle_rad)
        ])

        for dim in self.hidden_dims:
            x_norm = nn.gelu(nn.LayerNorm()(nn.Dense(dim)(x_norm)))
        return nn.Dense(3)(x_norm)


# ============================================================================
# PHYSICS MODEL CLASS
# ============================================================================

class PhysicsModel:
    """Pure JAX physics model - replaces Tesseract containers."""
    
    def __init__(self, model: nn.Module, params: dict, name: str):
        self.model = model
        self.params = params
        self.name = name
        self._forward = jit(lambda p, x: model.apply(p, x))
    
    def __call__(self, notch_angle: float, reynolds_number: float, roughness: float) -> jnp.ndarray:
        """Returns [drag, lift, side] forces in Newtons."""
        inputs = jnp.array([roughness, notch_angle, reynolds_number])
        return self._forward(self.params, inputs)
    
    @classmethod
    def load(cls, backend: str = "jaxphysics", weights_dir: Optional[Path] = None) -> "PhysicsModel":
        """Load a pre-trained physics model."""
        if weights_dir is None:
            weights_dir = Path(__file__).parent / "weights"
        
        weights_path = weights_dir / f"{backend}_weights.msgpack"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found at {weights_path}")
        
        model = JaxPhysicsNetwork() if backend == "jaxphysics" else SimplePhysicsNetwork()
        dummy_params = model.init(jax.random.PRNGKey(0), jnp.ones(3))
        
        with open(weights_path, "rb") as f:
            params = flax.serialization.from_bytes(dummy_params, f.read())
        
        print(f"✓ Loaded {backend} model")
        return cls(model, params, backend)


# Global cache
_MODEL_CACHE = {}

def get_physics_model(backend: str = "jaxphysics") -> PhysicsModel:
    """Get a cached physics model instance."""
    if backend not in _MODEL_CACHE:
        _MODEL_CACHE[backend] = PhysicsModel.load(backend)
    return _MODEL_CACHE[backend]


@jit
def analytical_physics(notch_angle: float, reynolds_number: float, roughness: float) -> jnp.ndarray:
    """Analytical model based on Mehta (1985) - no weights needed."""
    diameter, rho, mu = 0.072, 1.225, 1.81e-5
    area = jnp.pi * (diameter/2)**2
    
    rough_norm = jnp.clip(roughness, 0.0, 1.0)
    angle_rad = jnp.deg2rad(jnp.clip(notch_angle, -30.0, 30.0))
    re = jnp.clip(reynolds_number, 5e4, 3e5)
    
    velocity = re * mu / (rho * diameter)
    q = 0.5 * rho * velocity**2
    
    # Drag
    re_crit = 2.5e5 - 1.3e5 * rough_norm
    transition = jax.nn.sigmoid(3.0 / re_crit * (re - re_crit))
    cd = 0.50 - 0.30 * transition
    drag = cd * (1.0 + 0.15 * rough_norm * jnp.cos(angle_rad)**2) * q * area
    
    # Lift
    lift = 0.05 * rough_norm * jnp.sin(2.0 * angle_rad) * q * area
    
    # Swing
    optimal_angle = jnp.deg2rad(20.0)
    angle_eff = jnp.sin(angle_rad) * jnp.exp(-((angle_rad - optimal_angle)**2) / 0.3)
    conv_strength = jnp.exp(-((re - 1.3e5)**2) / (6e4)**2)
    rev_strength = jax.nn.sigmoid(5.0 * (re - 2.0e5) / 2.0e5)
    cs = 0.20 * conv_strength * angle_eff * rough_norm * (1 - rev_strength)
    side = cs * q * area
    
    return jnp.array([jnp.clip(drag, 0.1, 2.0), jnp.clip(lift, -0.3, 0.3), jnp.clip(side, -0.4, 0.4)])

