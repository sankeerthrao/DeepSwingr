#!/bin/bash
set -e

echo "Starting training for all tesseracts..."

# Train jaxphysics
if [ -d "tesseracts/jaxphysics" ]; then
    echo "Training jaxphysics..."
    cd tesseracts/jaxphysics
    python3 train_jaxphysics.py
    cd ../..
fi

# Train simplephysics
if [ -d "tesseracts/simplephysics" ]; then
    echo "Training simplephysics..."
    cd tesseracts/simplephysics
    python3 train_simplephysics.py
    cd ../..
fi

echo "Training complete!"

