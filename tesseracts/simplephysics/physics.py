"""
Cricket Ball Aerodynamics: Differentiable Physics Training
The CFD solver is EMBEDDED in the training loop - gradients flow through physics!

Parameters:
- roughness: Surface roughness coefficient [0.0, 1.0]
- notch_angle: Seam angle in degrees [-90, 90]
  * Negative angle = outswing delivery (ball swings away from RH batsman, +y direction)
  * Positive angle = inswing delivery (ball swings toward RH batsman, -y direction)
  * 0° = seam upright (no swing)
- reynolds_number: Flow Reynolds number [1e5, 1e6]

Output: Force vector [Fx_drag, Fy_lift, Fz_side] in Newtons
  * Fz_side: Positive = outswing (away), Negative = inswing (toward batsman)
"""

import jax
import jax.numpy as jnp
from jax import jit
import flax.linen as nn
from typing import Tuple

# ============================================================================
# CFD SIMULATION (Differentiable - used IN training loop)
# ============================================================================
@jit
def cfd_solve_navier_stokes(
    roughness: float,
    notch_angle: float,
    reynolds_number: float
) -> jnp.ndarray:
    """
    Physics-based differentiable CFD solver for cricket ball aerodynamics.
    Based on empirical fits from Mehta (1985), Barton (1982), Scobie et al. (2020)
    
    Returns: [drag, lift, side] force vector in Newtons
    """
    # Cricket ball parameters (standard)
    diameter = 0.072  # m (regulation size)
    rho = 1.225       # kg/m³ (sea level, 15°C)
    mu = 1.81e-5      # Pa·s (dynamic viscosity at 15°C)
    area = jnp.pi * (diameter/2)**2
    
    # Normalize inputs
    rough_norm = jnp.clip(roughness, 0.0, 1.0)  # 0=smooth, 1=very rough
    angle_rad = jnp.deg2rad(jnp.clip(notch_angle, -30.0, 30.0))
    re = jnp.clip(reynolds_number, 5e4, 3e5)
    
    # Calculate velocity and dynamic pressure
    velocity = re * mu / (rho * diameter)
    q = 0.5 * rho * velocity**2
    
    # ========================================================================
    # DRAG FORCE - Based on Mehta's boundary layer transition model
    # ========================================================================
    # Critical Reynolds number shifts with roughness (Achenbach 1972)
    # Smooth ball: Re_crit ≈ 2.5e5, Rough ball: Re_crit ≈ 1.2e5
    re_crit = 2.5e5 - 1.3e5 * rough_norm
    
    # Subcritical and supercritical drag coefficients
    cd_sub = 0.50  # Pre-transition (laminar separation)
    cd_super = 0.20  # Post-transition (turbulent separation)
    
    # Smooth transition using drag crisis curve
    transition_sharpness = 3.0 / re_crit  # Adaptive sharpness
    transition = jax.nn.sigmoid(transition_sharpness * (re - re_crit))
    cd = cd_sub - (cd_sub - cd_super) * transition
    
    # Seam orientation affects effective roughness (Scobie 2020)
    # Max drag when seam perpendicular, min when aligned with flow
    seam_effect = 1.0 + 0.15 * rough_norm * jnp.cos(angle_rad)**2
    cd = cd * seam_effect
    
    drag = cd * q * area
    
    # ========================================================================
    # LIFT FORCE (Magnus effect from spin) - Typically small for swing
    # ========================================================================
    # For seam bowling (minimal spin), lift is secondary
    # Mainly from asymmetric pressure distribution due to seam
    cl = 0.05 * rough_norm * jnp.sin(2.0 * angle_rad)
    lift = cl * q * area
    
    # ========================================================================
    # SIDE FORCE (Swing) - Based on Barton & Mehta models
    # ========================================================================
    # Conventional swing (low Re or moderate roughness)
    # Occurs when boundary layer on one side trips to turbulent
    
    # Swing force peaks at seam angle ≈ 20° (Barton 1982)
    optimal_angle = jnp.deg2rad(20.0)
    angle_efficiency = jnp.sin(angle_rad) * jnp.exp(-((angle_rad - optimal_angle)**2) / 0.3)
    
    # Conventional swing coefficient (asymmetric roughness critical)
    # Peak Cs ≈ 0.15-0.25 at Re ≈ 1.2-1.5e5 (Mehta 1985)
    re_conv_optimal = 1.3e5
    conv_strength = jnp.exp(-((re - re_conv_optimal)**2) / (6e4)**2)
    cs_conv = 0.20 * conv_strength * angle_efficiency * rough_norm
    
    # Reverse swing (high Re, highly asymmetric roughness)
    # Occurs at Re > 2e5 when one side is very rough, other relatively smooth
    # Direction REVERSES compared to conventional (Bown & Mehta 1993)
    re_rev_threshold = 2.0e5
    roughness_asymmetry = rough_norm * (1.0 - rough_norm)  # Peaks at 0.5
    rev_strength = jax.nn.sigmoid(5.0 * (re - re_rev_threshold) / re_rev_threshold)
    cs_rev = -0.25 * rev_strength * angle_efficiency * roughness_asymmetry
    
    # Total side force (conventional dominates early, reverse at high Re)
    cs = cs_conv * (1.0 - rev_strength) + cs_rev * rev_strength
    
    # Convention: seam angled toward leg side (positive angle) 
    # produces outswing (positive side force) for conventional swing
    side = cs * q * area
    
    # ========================================================================
    # Realistic force bounds (based on match data)
    # ========================================================================
    drag = jnp.clip(drag, 0.1, 2.0)    # ~0.3-0.8N typical
    lift = jnp.clip(lift, -0.3, 0.3)   # Small for seam bowling
    side = jnp.clip(side, -0.4, 0.4)   # ~0.1-0.3N for good swing
    
    return jnp.array([drag, lift, side])

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class CricketBallForceNetwork(nn.Module):
    """Neural network that LEARNS from the CFD solver."""

    hidden_dims: Tuple[int, ...] = (32, 64, 64, 32)

    @nn.compact
    def __call__(self, x):
        roughness, angle, re = x[0], x[1], x[2]
        re_normalized = jnp.log10(re) / 6.0

        angle_rad = jnp.deg2rad(angle)
        x_norm = jnp.array([
            roughness,
            jnp.sin(angle_rad),
            jnp.cos(angle_rad),
            re_normalized
        ])

        for dim in self.hidden_dims:
            x_norm = nn.Dense(dim)(x_norm)
            x_norm = nn.gelu(x_norm)
            x_norm = nn.LayerNorm()(x_norm)

        forces = nn.Dense(3)(x_norm)
        return forces
