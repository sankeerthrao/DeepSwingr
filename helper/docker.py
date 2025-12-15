"""
Docker environment setup and management for tesseract containers
"""
import subprocess
import sys
import time


def check_docker_network(network_name: str) -> bool:
    """Check if a Docker network exists."""
    try:
        result = subprocess.run(
            ["docker", "network", "ls", "--filter",
                f"name={network_name}", "--format", "{{.Name}}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        networks = result.stdout.strip().split("\\n")
        return network_name in networks
    except Exception as e:
        print(f"Failed to check Docker networks: {e}")
        return False


def ensure_docker_network(network_name: str) -> bool:
    """Create a Docker network if it does not exist."""
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
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def ensure_simplephysics_container(network_name: str):
    """Start simplephysics container if not already running"""
    print("\n🔍 Checking simplephysics container...")

    # Check if already running
    check = subprocess.run(
        ["docker", "ps", "--filter", "name=simplephysics",
            "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    if "simplephysics" in check.stdout:
        print("✓ simplephysics is already running")
        return

    print("  Cleaning up old simplephysics container if any...")
    subprocess.run(["docker", "stop", "simplephysics"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "rm", "simplephysics"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        print(f"❌ Failed to start simplephysics container:")
        print(result.stderr)
        raise RuntimeError("Failed to start simplephysics container")

    print("✓ simplephysics container started")

    # Wait for it to be ready
    print("  Waiting for simplephysics to be ready...")
    max_wait = 30
    for i in range(max_wait):
        time.sleep(1)
        check = subprocess.run(
            ["docker", "ps", "--filter", "name=simplephysics",
                "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        if "simplephysics" not in check.stdout:
            print("❌ simplephysics container stopped unexpectedly")
            logs = subprocess.run(
                ["docker", "logs", "simplephysics"], capture_output=True, text=True)
            print(logs.stdout)
            print(logs.stderr)
            raise RuntimeError("simplephysics container stopped unexpectedly")

        health_check = subprocess.run(
            ["curl", "-f", "http://localhost:8000/health"],
            capture_output=True,
            text=True,
        )
        if health_check.returncode == 0:
            print(f"✓ simplephysics is ready (took {i+1}s)")
            break
        if i == max_wait - 1:
            print("⚠️  Warning: simplephysics may not be fully ready")
            logs = subprocess.run(
                ["docker", "logs", "simplephysics"], capture_output=True, text=True)
            print(logs.stdout[-500:])


def prepare_docker_environment(network_name: str):
    """Set up Docker environment including network and containers"""
    print(f"============================================================")
    print(f"  CRICKET BALL TRAJECTORY SIMULATION DEMO")
    print(f"============================================================")

    ensure_docker_network(network_name)
    ensure_simplephysics_container(network_name)


def cleanup_containers():
    """Stop and remove containers"""
    print("\n🧹 Cleaning up all containers...")
    subprocess.run(["docker", "stop", "simplephysics"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "rm", "simplephysics"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✓ Cleanup complete")
