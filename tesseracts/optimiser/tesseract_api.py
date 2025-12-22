# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tesseract API module for optimizer tesseract
Inputs: fixed_variables, optimization_variables, swing_type, swing_url, physics_url, integrator_url
Outputs: optimal_parameters, maximum_deviation, swing_possible
"""
from typing import Any, Dict, List, Literal
import numpy as np
import jax
import jax.numpy as jnp
import optax
from pydantic import BaseModel, Field, model_validator

from tesseract_core.runtime import Float32


class InputSchema(BaseModel):
    """Input schema for optimizer tesseract"""

    fixed_variables: Dict[str, Float32] = Field(
        ..., description="Fixed parameters that don't change during optimization")
    optimization_variables: Dict[str, List[Float32]] = Field(
        ..., description="Variables to optimize with [min, max] bounds")
    swing_type: Literal["in", "out", "reverse"] = Field(
        ..., description="Type of swing to optimize for")
    swing_url: str = Field(
        default="http://swing:8000",
        description="URL of swing tesseract")
    physics_url: str = Field(
        default="http://simplephysics:8000",
        description="URL of physics backend")
    integrator_url: str = Field(
        default="http://integrator:8000",
        description="URL of integrator tesseract")

    @model_validator(mode="after")
    def validate_inputs(self):
        # Validate that optimization variables have valid bounds
        for var_name, bounds in self.optimization_variables.items():
            if len(bounds) != 2:
                raise ValueError(f"optimization_variables[{var_name}] must have exactly 2 bounds [min, max]")
            if bounds[0] >= bounds[1]:
                raise ValueError(f"optimization_variables[{var_name}] min ({bounds[0]}) must be < max ({bounds[1]})")

        # Validate that swing_type is valid
        if self.swing_type not in ["in", "out", "reverse"]:
            raise ValueError(f"swing_type must be 'in', 'out', or 'reverse'. Got {self.swing_type}")

        return self


class OutputSchema(BaseModel):
    """Output schema for optimizer tesseract"""

    optimal_parameters: Dict[str, Float32] = Field(
        ..., description="Optimal values for the optimization variables")
    maximum_deviation: Float32 = Field(
        ..., description="Maximum swing deviation achieved (cm)")
    swing_possible: bool = Field(
        ..., description="Whether swing is possible in the given range (especially for reverse swing)")


def create_objective_function(fixed_vars: Dict[str, float], swing_tesseract, swing_type: str,
                             physics_url: str, integrator_url: str):
    """Create objective function for optimization"""

    def objective(params_dict: Dict[str, float]) -> float:
        """Objective function that combines fixed and variable parameters"""

        # Combine fixed and variable parameters
        all_params = {**fixed_vars, **params_dict}

        # Call swing tesseract with required URLs
        try:
            result = swing_tesseract.apply({
                "initial_velocity": all_params["initial_velocity"],
                "release_angle": all_params["release_angle"],
                "roughness": all_params["roughness"],
                "seam_angle": all_params["seam_angle"],
                "physics_url": physics_url,
                "integrator_url": integrator_url
            })
            deviation = result['maximum_deviation']

            # For "in" swing, we want negative deviation (towards batsman)
            # For "out" swing, we want positive deviation (away from batsman)
            # For "reverse" swing, we want maximum absolute deviation
            if swing_type == "in":
                return -deviation  # Minimize negative deviation (maximize towards batsman)
            elif swing_type == "out":
                return deviation   # Maximize positive deviation (away from batsman)
            else:  # reverse
                return -abs(deviation)  # Maximize absolute deviation

        except Exception as e:
            # Return large penalty for invalid configurations
            return 1000.0

    return objective


def optimize_swing(inputs: InputSchema) -> OutputSchema:
    """Optimize swing parameters using JAX"""

    # Connect to swing tesseract
    from tesseract_core import Tesseract
    swing_tesseract = Tesseract.from_url(inputs.swing_url)

    # Get optimization variable names and bounds
    opt_vars = list(inputs.optimization_variables.keys())
    bounds = [inputs.optimization_variables[var] for var in opt_vars]

    # Convert bounds to JAX arrays
    bounds_array = jnp.array(bounds)

    # Create objective function
    objective = create_objective_function(
        inputs.fixed_variables,
        swing_tesseract,
        inputs.swing_type,
        inputs.physics_url,
        inputs.integrator_url
    )

    # JAX-compatible objective for optimization
    def jax_objective(x):
        params_dict = {opt_vars[i]: float(x[i]) for i in range(len(opt_vars))}
        return objective(params_dict)

    # Use differential evolution for global optimization (works well for bounded problems)
    from scipy.optimize import differential_evolution

    # Define bounds for scipy
    scipy_bounds = [(float(b[0]), float(b[1])) for b in bounds]

    # Run optimization
    result = differential_evolution(
        jax_objective,
        bounds=scipy_bounds,
        maxiter=50,
        popsize=10,
        seed=42
    )

    # Extract optimal parameters
    optimal_params = {opt_vars[i]: float(result.x[i]) for i in range(len(opt_vars))}
    max_deviation = -result.fun  # Convert back from minimization to maximization

    # Check if swing is possible (especially for reverse swing)
    # If the maximum deviation is very small, swing might not be possible
    swing_possible = abs(max_deviation) > 0.1  # Threshold of 0.1 cm

    return OutputSchema(
        optimal_parameters=optimal_params,
        maximum_deviation=max_deviation,
        swing_possible=swing_possible
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply the optimizer to find optimal swing parameters"""
    return optimize_swing(inputs)
