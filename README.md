# Cricket Ball Swing Simulator

A modular, differentiable simulation of cricket ball trajectory and swing using [Tesseract Core](https://github.com/pasteurlabs/tesseract-core) and [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax).

## System Architecture

![System Architecture](assets/deepswingr_architecture.png)

The project follows a highly modular architecture where different physical components are isolated into independent **Tesseracts**:

1.  **Physics Backend (`simplephysics` / `jaxphysics`)**: Computes forces (drag, lift, side) acting on the ball based on its state (velocity, spin, seam angle, roughness).
2.  **Integrator (`integrator`)**: Performs numerical integration of the equations of motion to predict the ball's trajectory.
3.  **Swing Logic (`swing`)**: A high-level tesseract that orchestrates the integrator and physics backend to determine the final deviation of the ball.
4.  **Optimiser (`optimiser`)**: Searches for optimal parameters (e.g., the best seam angle for maximum swing) by interacting with the `swing` tesseract.

## Key Features

- **Modular Design**: Each component is a containerized Tesseract, allowing for easy swapping (e.g., replacing `simplephysics` with a neural-network-based `jaxphysics`).
- **Differentiable Programming**: By using JAX, Flax, and Equinox, the `jaxphysics` tesseract provides automatic differentiation (AD) endpoints. This allows for obtaining exact **Jacobians** via the Tesseract SDK, enabling gradient-based optimization and sensitivity analysis.
- **Apache 2.0 Licensed**: All core libraries and this template are released under the Apache License 2.0, ensuring freedom for both academic and commercial use.

## Get Started

### Prerequisites

- Python 3.10+
- Docker (Desktop recommended)
- `pip install -r requirements.txt`

### Steps to Reproduce

1.  **Clone and Setup**:

    ```bash
    git clone <your-repo-url>
    cd tesseract-hackathon-template
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Train Physics Models**:
    Run the training script to generate weights for the neural network physics engines:

    ```bash
    ./trainall.sh
    ```

3.  **Build Tesseracts**:
    Build the Docker images for all tesseracts:

    ```bash
    ./buildall.sh
    ```

4.  **Run the Simulator**:
    Launch the interactive CLI to visualize trajectories or optimize swing:
    ```bash
    python main.py
    ```

## Differentiable Pipeline

The `simplephysics` and `jaxphysics` tesseracts implement standard AD endpoints:

- `jacobian`: Full Jacobian matrix for sensitivity analysis.
- `jacobian_vector_product` (JVP): Forward-mode AD.
- `vector_jacobian_product` (VJP): Reverse-mode AD.

These can be called via the Tesseract SDK:

```python
from tesseract_core import Tesseract

physics = Tesseract.from_url("http://localhost:8001")
inputs = {"notch_angle": 30.0, "reynolds_number": 5e5, "roughness": 0.8}
jac = physics.jacobian(inputs, jac_inputs={"notch_angle"}, jac_outputs={"force_vector"})
print(f"Sensitivity to seam angle: {jac}")
```

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

## Contact

Please use the [issues tracker](https://github.com/gpavanb1/deepswingr/issues) for any questions.
