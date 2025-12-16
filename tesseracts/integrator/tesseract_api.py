"""
Tesseract API module for integrator tesseract
Inputs: initial_velocity, release_angle, roughness, seam_angle, physics_url
Outputs: trajectory data (times, x, y, z positions, velocities)
"""
from typing import Any
from pydantic import BaseModel, Field
from typing_extensions import Self
from tesseract_core.runtime import Array, Float32
from integrate import simulate_trajectory


class InputSchema(BaseModel):
    """Input schema for integrator tesseract"""
    initial_velocity: Float32 = Field(
        ..., description="Initial velocity in m/s [20.0, 50.0]")
    release_angle: Float32 = Field(
        ..., description="Release angle in degrees [0.0, 10.0]")
    roughness: Float32 = Field(
        ..., description="Ball surface roughness [0.0, 1.0]")
    seam_angle: Float32 = Field(
        ..., description="Seam angle in degrees [-90.0, 90.0]")
    physics_url: str = Field(
        default="http://simplephysics:8000",
        description="URL of physics backend (simplephysics or jaxphysics)"
    )


class OutputSchema(BaseModel):
    """Output schema for integrator tesseract"""
    times: Array[(None,), Float32] = Field(
        ..., description="Time points of trajectory")
    x_positions: Array[(None,), Float32] = Field(
        ..., description="X positions along trajectory")
    y_positions: Array[(None,), Float32] = Field(
        ..., description="Y positions along trajectory")
    z_positions: Array[(None,), Float32] = Field(
        ..., description="Z positions along trajectory")
    velocities: Array[(None, 3), Float32] = Field(
        ..., description="Velocity vectors along trajectory")


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply the integrator to get ball trajectory"""
    # Run simulation with specified physics backend
    times, x, y, z, velocities = simulate_trajectory(
        initial_velocity=inputs.initial_velocity,
        release_angle=inputs.release_angle,
        roughness=inputs.roughness,
        seam_angle=inputs.seam_angle,
        physics_url=inputs.physics_url  # Pass physics_url to simulation
    )

    # Return trajectory data
    return OutputSchema(
        times=times,
        x_positions=x,
        y_positions=y,
        z_positions=z,
        velocities=velocities
    )
