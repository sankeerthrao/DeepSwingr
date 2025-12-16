"""
Main Pipeline - Tesseract Hackathon Template

This script demonstrates usage of the integrator tesseract which can talk to
both simplephysics and jaxphysics backends for trajectory simulation.
"""
from helper.docker import prepare_docker_environment, cleanup_containers
from tesseract_core import Tesseract
from plotter.plt import plot_trajectory_3d, animate_trajectory, compare_deliveries
from plotter.interactive import scenarios


def integrator_tesseract_demo():
    network_name = "tesseract_network"

    # Start both physics backends (simplephysics on 8000, jaxphysics on 8001)
    prepare_docker_environment(network_name, use_jaxphysics=True)

    print(f"\n🚀 Starting integrator...")

    # Start integrator that will call physics backends
    integrator = Tesseract.from_image(
        "integrator",
        network=network_name,
        network_alias="integrator"
    )

    try:
        with integrator:
            print("✓ Integrator started\n")

            # Store results for comparison
            all_results = []

            for key, scenario in scenarios.items():
                if key == "5":
                    continue

                print(f"\n{'='*60}")
                print(f"Simulating: {scenario['name']}")
                print(f"{'='*60}")
                print(f"  Speed: {scenario['velocity']*3.6:.1f} km/h")
                print(f"  Release angle: {scenario['angle']}°")
                print(f"  Roughness: {scenario['roughness']}")
                print(f"  Seam angle: {scenario['seam_angle']}°")

                params = {
                    "initial_velocity": scenario["velocity"],
                    "release_angle": scenario["angle"],
                    "roughness": scenario["roughness"],
                    "seam_angle": scenario["seam_angle"],
                }

                simple_result = None
                jax_result = None

                # Run with simplephysics backend
                try:
                    print("  🔄 Running with simplephysics backend...")
                    params_simple = {
                        **params, "physics_url": "http://simplephysics:8000"}
                    simple_result = integrator.apply(params_simple)
                    print("  ✓ Simplephysics simulation complete")
                except Exception as e:
                    print(f"  ❌ Simplephysics error: {e}")

                # Run with jaxphysics backend
                try:
                    print("  🔄 Running with jaxphysics backend...")
                    params_jax = {
                        **params, "physics_url": "http://jaxphysics:8000"}
                    jax_result = integrator.apply(params_jax)
                    print("  ✓ Jaxphysics simulation complete")
                except Exception as e:
                    print(f"  ❌ Jaxphysics error: {e}")

                # Generate plot with both if available
                if simple_result is not None or jax_result is not None:
                    print("  📊 Generating plots...")
                    try:
                        # Build trajectory data for both physics engines
                        trajectories = []

                        if simple_result is not None:
                            trajectories.append({
                                "name": "Simplephysics",
                                "times": simple_result["times"],
                                "x": simple_result["x_positions"],
                                "y": simple_result["y_positions"],
                                "z": simple_result["z_positions"],
                                "velocities": simple_result.get("velocities", None)
                            })

                        if jax_result is not None:
                            trajectories.append({
                                "name": "Jaxphysics (CFD)",
                                "times": jax_result["times"],
                                "x": jax_result["x_positions"],
                                "y": jax_result["y_positions"],
                                "z": jax_result["z_positions"],
                                "velocities": jax_result.get("velocities", None)
                            })

                        # Use first available result for backward compatibility with old plot API
                        primary = simple_result if simple_result is not None else jax_result

                        # 3D trajectory plot
                        fig_3d = plot_trajectory_3d(
                            primary["times"],
                            primary["x_positions"],
                            primary["y_positions"],
                            primary["z_positions"],
                            primary.get("velocities"),
                            scenario["velocity"],
                            scenario["roughness"],
                            scenario["seam_angle"],
                            use_plotly=True,
                            # Pass both trajectories for comparison if plot supports it
                            all_trajectories=trajectories
                        )
                        fig_3d.show()
                        print("  ✓ Plot displayed")
                    except Exception as e:
                        print(f"  ⚠️  Plot error: {e}")

                    # Store results for comparison
                    all_results.append({
                        "name": scenario["name"],
                        "velocity": scenario["velocity"],
                        "angle": scenario["angle"],
                        "roughness": scenario["roughness"],
                        "seam_angle": scenario["seam_angle"],
                        "simple": simple_result,
                        "jax": jax_result
                    })

            # Show comparison of all deliveries
            if len(all_results) > 1:
                print("\n📊 Generating delivery comparison...")

                # Prepare data for compare_deliveries
                # Format: list of (name, velocity, angle, roughness, seam_angle, times, x, y, z)
                comparison_data = []

                for r in all_results:
                    if r["simple"] is not None:
                        comparison_data.append((
                            f"{r['name']} (Simple)",
                            r["velocity"],
                            r["angle"],
                            r["roughness"],
                            r["seam_angle"],
                            r["simple"]["times"],
                            r["simple"]["x_positions"],
                            r["simple"]["y_positions"],
                            r["simple"]["z_positions"]
                        ))

                    if r["jax"] is not None:
                        comparison_data.append((
                            f"{r['name']} (JAX-CFD)",
                            r["velocity"],
                            r["angle"],
                            r["roughness"],
                            r["seam_angle"],
                            r["jax"]["times"],
                            r["jax"]["x_positions"],
                            r["jax"]["y_positions"],
                            r["jax"]["z_positions"]
                        ))

                if comparison_data:
                    compare_deliveries(comparison_data)
                    print("  ✓ Comparison displayed")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Cleaning up containers...")
        cleanup_containers()


def main() -> None:
    try:
        integrator_tesseract_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")


if __name__ == "__main__":
    main()
