"""
Docker environment setup and management for tesseract containers
"""
import subprocess
import sys
import time
from tesseract_core import Tesseract


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


def ensure_container(container_name: str, image_name: str,
                     network_name: str, host_port: int):
    """Start a container if not already running"""
    print(f"\n🔍 Checking {container_name} container...")

    # Check if already running
    check = subprocess.run(
        ["docker", "ps", "--filter", f"name={container_name}",
            "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    if container_name in check.stdout:
        print(f"✓ {container_name} is already running")
        return

    print(f"  Cleaning up old {container_name} container if any...")
    subprocess.run(["docker", "stop", container_name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "rm", container_name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"  Starting {container_name} container on port {host_port}...")
    result = subprocess.run([
        "docker", "run", "-d",
        "--name", container_name,
        "--network", network_name,
        "--network-alias", container_name,
        "-p", f"{host_port}:8000",  # Map host_port to container's 8000
        image_name,
        "serve", "--host", "0.0.0.0", "--port", "8000"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Failed to start {container_name} container:")
        print(result.stderr)
        raise RuntimeError(f"Failed to start {container_name} container")

    print(f"✓ {container_name} container started")

    # Wait for it to be ready
    print(f"  Waiting for {container_name} to be ready...")
    max_wait = 30
    for i in range(max_wait):
        time.sleep(1)
        check = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}",
                "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        if container_name not in check.stdout:
            print(f"❌ {container_name} container stopped unexpectedly")
            logs = subprocess.run(
                ["docker", "logs", container_name], capture_output=True, text=True)
            print(logs.stdout)
            print(logs.stderr)
            raise RuntimeError(
                f"{container_name} container stopped unexpectedly")

        health_check = subprocess.run(
            ["curl", "-f", f"http://localhost:{host_port}/health"],
            capture_output=True,
            text=True,
        )
        if health_check.returncode == 0:
            print(f"✓ {container_name} is ready (took {i+1}s)")
            break
        if i == max_wait - 1:
            print(f"⚠️  Warning: {container_name} may not be fully ready")
            logs = subprocess.run(
                ["docker", "logs", container_name], capture_output=True, text=True)
            print(logs.stdout[-500:] if logs.stdout else "No logs")


# Configuration for physics backends
PHYSICS_BACKENDS = {
    "simplephysics": {
        "container": "simplephysics",
        "image": "simplephysics",
        "port": 8000,
        "url": "http://simplephysics:8000"
    },
    "jaxphysics": {
        "container": "jaxphysics",
        "image": "jaxphysics",
        "port": 8001,
        "url": "http://jaxphysics:8000"
    }
}


def ensure_physics_container(backend_name: str, network_name: str):
    """Start the specified physics backend container if not already running"""
    if backend_name not in PHYSICS_BACKENDS:
        raise ValueError(f"Unknown physics backend: {backend_name}")
    
    config = PHYSICS_BACKENDS[backend_name]
    ensure_container(
        config["container"], 
        config["image"], 
        network_name, 
        config["port"]
    )


def prepare_docker_environment(network_name: str, backend_name: str = "simplephysics"):
    """Set up Docker environment including network and containers"""
    print("============================================================")
    print("  CRICKET BALL TRAJECTORY SIMULATION DEMO")
    print("============================================================")

    ensure_docker_network(network_name)
    
    # Let's just ensure the requested one.
    ensure_physics_container(backend_name, network_name)


def cleanup_containers():
    """Stop and remove containers"""
    global _INTEGRATOR, _SWING, _OPTIMIZER
    print("\n🧹 Cleaning up all containers...")
    
    # Reset globals
    _INTEGRATOR = None
    _SWING = None
    _OPTIMIZER = None

    containers = ["simplephysics", "jaxphysics", "integrator", "swing", "optimiser"]
    for container in containers:
        subprocess.run(["docker", "stop", container],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["docker", "rm", container],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("✓ Cleanup complete")


def switch_backend(backend_name: str, network_name: str = "tesseract_network"):
    """Switch the physics backend by resetting tesseracts and ensuring container."""
    global _INTEGRATOR, _SWING, _OPTIMIZER
    
    if backend_name not in PHYSICS_BACKENDS:
        raise ValueError(f"Unknown physics backend: {backend_name}")
    
    print(f"\n🔄 Switching physics backend to: {backend_name}")
    
    # Reset globals to force re-setup on next get_tesseracts() call
    _INTEGRATOR = None
    _SWING = None
    _OPTIMIZER = None
    
    # Ensure the new container is running
    ensure_physics_container(backend_name, network_name)
    
    return backend_name


# Global tesseracts to avoid restarting containers on every call
_INTEGRATOR = None
_SWING = None
_OPTIMIZER = None


def get_tesseracts(network_name="tesseract_network", backend_name="simplephysics"):
    """Get or setup tesseracts (singleton pattern)"""
    global _INTEGRATOR, _SWING, _OPTIMIZER
    if _INTEGRATOR is None:
        print("🔍 Starting Tesseracts (first time setup)...")
        _INTEGRATOR, _SWING, _OPTIMIZER = setup_tesseracts(network_name, backend_name)
    return _INTEGRATOR, _SWING, _OPTIMIZER


def setup_tesseracts(network_name="tesseract_network", backend_name="simplephysics"):
    """Setup all tesseracts (integrator, swing, optimizer)"""
    print(f"🚀 Setting up tesseracts with {backend_name} backend...")

    # Start physics backends
    prepare_docker_environment(network_name, backend_name=backend_name)

    # Start tesseracts with external port mappings
    ensure_container("integrator", "integrator", network_name, 8002)
    ensure_container("swing", "swing", network_name, 8003)
    ensure_container("optimiser", "optimiser", network_name, 8004)

    # Connect to running tesseracts
    integrator = Tesseract.from_url("http://localhost:8002")
    swing = Tesseract.from_url("http://localhost:8003")
    optimizer = Tesseract.from_url("http://localhost:8004")

    print("✓ Tesseracts ready")
    return integrator, swing, optimizer
