"""
Main Pipeline - Tesseract Hackathon Template
"""
from tesseract_core import Tesseract
from plotter.plt import plot_trajectory_3d
from plotter.interactive import scenarios
import subprocess
import sys
import time


def ensure_docker_network(network_name="tesseract_network"):
    """Ensure Docker network exists"""
    print(f"\n🔍 Checking Docker network '{network_name}'...")

    try:
        inspect_result = subprocess.run(
            ["docker", "network", "inspect", network_name],
            capture_output=True,
            text=True
        )

        if inspect_result.returncode != 0:
            print(f"   Creating network '{network_name}'...")
            subprocess.run(
                ["docker", "network", "create", network_name],
                check=True,
                capture_output=True
            )
            print(f"✓ Network '{network_name}' created")
        else:
            print(f"✓ Network '{network_name}' exists")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def start_simplephysics(network_name="tesseract_network"):
    """Start simplephysics container if not already running"""
    print("\n🔍 Checking simplephysics container...")

    # Check if already running
    check = subprocess.run(
        ["docker", "ps", "--filter", "name=simplephysics",
            "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if "simplephysics" in check.stdout:
        print("✓ simplephysics is already running")
        return

    # Stop and remove any existing stopped container
    print("  Cleaning up old simplephysics container...")
    subprocess.run(
        ["docker", "stop", "simplephysics"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    subprocess.run(
        ["docker", "rm", "simplephysics"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Start new container on port 8000 (matching simplephysics default)
    print("  Starting simplephysics container...")

    result = subprocess.run([
        "docker", "run", "-d",
        "--name", "simplephysics",
        "--network", network_name,
        "--network-alias", "simplephysics",
        "-p", "8000:8000",
        "simplephysics",
        "serve", "--host", "0.0.0.0", "--port", "8000"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Failed to start simplephysics:")
        print(result.stderr)
        sys.exit(1)

    print("✓ simplephysics container started")

    # Wait for it to be ready
    print("  Waiting for simplephysics to be ready...")
    max_wait = 30
    for i in range(max_wait):
        time.sleep(1)

        # Check if container is still running
        check = subprocess.run(
            ["docker", "ps", "--filter", "name=simplephysics",
                "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )

        if "simplephysics" not in check.stdout:
            print("❌ simplephysics container stopped unexpectedly")
            print("\nContainer logs:")
            logs = subprocess.run(["docker", "logs", "simplephysics"],
                                  capture_output=True, text=True)
            print(logs.stdout)
            print(logs.stderr)
            sys.exit(1)

        # Try to connect
        health_check = subprocess.run(
            ["curl", "-f", "http://localhost:8000/health"],
            capture_output=True,
            text=True
        )

        if health_check.returncode == 0:
            print(f"✓ simplephysics is ready (took {i+1}s)")
            break

        if i == max_wait - 1:
            print("⚠️  Warning: simplephysics may not be fully ready")
            print("\nContainer logs:")
            logs = subprocess.run(["docker", "logs", "simplephysics"],
                                  capture_output=True, text=True)
            print(logs.stdout[-500:])

    # Show connection info
    print("\n📡 Connection Info:")
    print(f"   Internal URL: http://simplephysics:8000")
    print(f"   External URL: http://localhost:8000")


def integrator_tesseract_demo():
    print("\n" + "=" * 60)
    print("  CRICKET BALL TRAJECTORY SIMULATION DEMO")
    print("=" * 60)

    network_name = "tesseract_network"

    # Setup
    ensure_docker_network(network_name)
    start_simplephysics(network_name)

    print(f"\n🚀 Starting integrator Tesseract...")

    integrator = Tesseract.from_image(
        "integrator",
        network=network_name,
        network_alias="integrator"
    )

    try:
        with integrator:
            print("✓ Integrator Tesseract started\n")

            for key, scenario in scenarios.items():
                if key == '5':
                    continue

                print(f"\n{'='*60}")
                print(f"Simulating: {scenario['name']}")
                print(f"{'='*60}")
                print(f"  Speed: {scenario['velocity']*3.6:.1f} km/h")
                print(f"  Release angle: {scenario['angle']}°")
                print(f"  Roughness: {scenario['roughness']}")
                print(f"  Seam angle: {scenario['seam_angle']}°")

                params = {
                    "initial_velocity": scenario['velocity'],
                    "release_angle": scenario['angle'],
                    "roughness": scenario['roughness'],
                    "seam_angle": scenario['seam_angle']
                }

                try:
                    print("  🔄 Running simulation...")
                    result = integrator.apply(params)
                    print("  ✓ Simulation complete")

                    print("  📊 Generating plot...")
                    fig = plot_trajectory_3d(
                        result['times'],
                        result['x_positions'],
                        result['y_positions'],
                        result['z_positions'],
                        result['velocities'],
                        scenario['velocity'],
                        scenario['roughness'],
                        scenario['seam_angle'],
                        use_plotly=True
                    )

                    fig.show()
                    print("  ✓ Plot displayed")

                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    # Show simplified error
                    if "ConnectionError" in str(e) or "Connection refused" in str(e):
                        print(
                            "  💡 Tip: simplephysics may not be responding. Check logs:")
                        print("     docker logs simplephysics")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Cleaning up integrator...")


def cleanup():
    """Stop and remove containers"""
    print("\n🧹 Cleaning up all containers...")
    subprocess.run(["docker", "stop", "simplephysics"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "rm", "simplephysics"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✓ Cleanup complete")


def main() -> None:
    try:
        integrator_tesseract_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        # Uncomment to auto-cleanup:
        # cleanup()
        pass


if __name__ == "__main__":
    main()
