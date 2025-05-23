#!/bin/bash
set -e

# Resolve script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Running setup from: $SCRIPT_DIR"

# Variables
ENV_NAME="agileflight"
UNITY_URL="https://download.ifi.uzh.ch/rpg/Flightmare/RPG_Flightmare.zip"
UNITY_DIR="$SCRIPT_DIR/sim/agile_flight/flightrender"
UNITY_ZIP="$UNITY_DIR/RPG_Flightmare.zip"

# 1. Create Conda environment
echo "Creating Conda environment: $ENV_NAME"
conda remove -n $ENV_NAME --all -y || true
conda create -n $ENV_NAME python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 2. Python dependencies
echo "Installing pip dependencies..."
pip install --upgrade pip setuptools wheel build
pip install -r "$SCRIPT_DIR/requirements.txt"

# 3. Build Flightmare (C++)
echo "Building Flightmare..."
cd "$SCRIPT_DIR/sim/flightmare/flightlib"
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd "$SCRIPT_DIR"

# 4. Install Flightgym (Python binding)
echo "Installing Flightgym Python module..."
pip install "$SCRIPT_DIR/sim/flightmare/flightlib"

# 5. Install RPG baseline wrapper
echo "Installing RPG wrapper..."
pip install "$SCRIPT_DIR/sim/flightmare/flightpy/flightrl"

# 6. Download Unity binary if needed
mkdir -p "$UNITY_DIR"
if [ ! -f "$UNITY_DIR/RPG_Flightmare/RPG_Flightmare.x86_64" ]; then
    echo "Downloading RPG_Flightmare Unity binary..."
    wget "$UNITY_URL" -O "$UNITY_ZIP"
    unzip "$UNITY_ZIP" -d "$UNITY_DIR"
    chmod +x "$UNITY_DIR/RPG_Flightmare/RPG_Flightmare.x86_64"
    rm "$UNITY_ZIP"
fi

# 7. Export FLIGHTMARE_PATH
if ! grep -q "FLIGHTMARE_PATH" ~/.bashrc; then
    echo "export FLIGHTMARE_PATH=$SCRIPT_DIR/sim/flightmare" >> ~/.bashrc
fi
export FLIGHTMARE_PATH="$SCRIPT_DIR/sim/flightmare"

echo "=== [OZIRES SETUP COMPLETE] ==="
echo "To run the simulator:"
echo "1. Open a terminal and run:"
echo "   conda activate $ENV_NAME"
echo "   $UNITY_DIR/RPG_Flightmare/RPG_Flightmare.x86_64"
echo ""
echo "2. In another terminal run training:"
echo "   conda activate $ENV_NAME"
echo "   python train.py --device cuda --episodes 1000"
