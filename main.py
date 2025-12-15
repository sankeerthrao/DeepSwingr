"""
Main Pipeline - Tesseract Hackathon Template

This script demonstrates usage of the integrator tesseract for cricket ball trajectory simulation.
"""
from helper.docker import prepare_docker_environment, cleanup_containers
from tesseract_core import Tesseract
from plotter.plt import plot_trajectory_3d, animate_trajectory, compare_deliveries
from plotter.interactive import scenarios


def integrator_tesseract_demo():
    network_name = "tesseract_network"

    prepare_docker_environment(network_name)

    print(f"\n🚀 Starting integrator Tesseract...")

    integrator = Tesseract.from_image(
        "integrator",
        network=network_name,
        network_alias="integrator"
    )

    try:
        with integrator:
            print("✓ Integrator Tesseract started\n")

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

                try:
                    print("  🔄 Running simulation...")
                    result = integrator.apply(params)
                    print("  ✓ Simulation complete")

                    print("  📊 Generating plots...")
                    # 3D trajectory plot
                    fig_3d = plot_trajectory_3d(
                        result["times"],
                        result["x_positions"],
                        result["y_positions"],
                        result["z_positions"],
                        result["velocities"],
                        scenario["velocity"],
                        scenario["roughness"],
                        scenario["seam_angle"],
                        use_plotly=True,
                    )
                    fig_3d.show()
                    print("  ✓ Plot displayed")

                    # Store results for comparison
                    all_results.append({
                        "name": scenario["name"],
                        "velocity": scenario["velocity"],
                        "angle": scenario["angle"],
                        "roughness": scenario["roughness"],
                        "seam_angle": scenario["seam_angle"],
                        "times": result["times"],
                        "x": result["x_positions"],
                        "y": result["y_positions"],
                        "z": result["z_positions"]
                    })

                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    if "ConnectionError" in str(e) or "Connection refused" in str(e):
                        print(
                            "  💡 Tip: simplephysics may not be responding. Check logs:"
                        )
                        print("     docker logs simplephysics")

            # Show comparison of all deliveries
            if len(all_results) > 1:
                print("\n📊 Generating delivery comparison...")
                # Use stored results for comparison
                compare_deliveries([
                    (r["name"], r["velocity"], r["angle"],
                     r["roughness"], r["seam_angle"],
                     r["times"], r["x"], r["y"], r["z"])
                    for r in all_results
                ])
                print("  ✓ Comparison displayed")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Cleaning up integrator...")
        cleanup_containers()


def main() -> None:
    try:
        integrator_tesseract_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")


if __name__ == "__main__":
    main()
