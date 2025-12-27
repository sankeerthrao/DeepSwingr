# 🚀 DeepSwingr Remote Deployment Options

No Tesseract needed! Here are simple ways to run DeepSwingr remotely:

## Option 1: Modal (Easiest - Serverless)

[Modal](https://modal.com) is the simplest way to run Python remotely. Free tier available.

```bash
# Install
pip install modal
modal token new  # One-time auth

# Run simulation on cloud
modal run modal_app.py

# Deploy as API (gets a public URL)
modal deploy modal_app.py
# Access: https://your-username--deepswingr-api.modal.run/simulate?velocity=35&seam_angle=30
```

**Why Modal?**
- No Docker needed (they handle it)
- Pay per second ($0.000025/sec for CPU)
- Auto-scaling
- Just decorate your functions with `@app.function()`

## Option 2: Docker + Any Cloud

```bash
# Build
cd deploy
docker build -t deepswingr -f Dockerfile ..

# Run locally
docker run -p 8000:8000 deepswingr

# Test
curl "http://localhost:8000/simulate?velocity=35&seam_angle=30"
```

### Deploy to AWS (ECS/Fargate)
```bash
# Push to ECR
aws ecr create-repository --repository-name deepswingr
docker tag deepswingr:latest YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/deepswingr:latest
docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/deepswingr:latest

# Create ECS service (or use AWS Copilot for simplicity)
copilot init --app deepswingr --name api --type "Load Balanced Web Service"
```

### Deploy to Google Cloud Run
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/deepswingr

# Deploy
gcloud run deploy deepswingr --image gcr.io/PROJECT_ID/deepswingr --allow-unauthenticated
```

### Deploy to Fly.io (Simple & Cheap)
```bash
flyctl launch  # Follow prompts
flyctl deploy
```

## Option 3: Ray (Distributed Computing)

For heavy batch jobs across multiple machines:

```python
import ray
ray.init()  # Connects to Ray cluster

@ray.remote
def simulate_batch(params_list):
    from simulator import compute_swing
    return [compute_swing(*p) for p in params_list]

# Run 1000 simulations in parallel
futures = [simulate_batch.remote(batch) for batch in batches]
results = ray.get(futures)
```

## Comparison

| Method | Setup Time | Cost | Best For |
|--------|-----------|------|----------|
| **Modal** | 5 min | ~$0.001/sim | Quick experiments, APIs |
| **Docker + Cloud Run** | 30 min | ~$0.0001/sim | Production APIs |
| **Ray** | 1 hour | Varies | Heavy batch processing |
| **Local** | 0 | Free | Development |

## The Key Insight

The neural network inference is **very lightweight** (~1ms per call). You don't need:
- Kubernetes
- Tesseract containers
- Complex orchestration

Just ship the weights file (138KB) and run anywhere!

