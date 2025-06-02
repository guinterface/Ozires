#!/bin/bash

# === Ozires Drone RL Setup Script ===

# --- Configurable Paths ---
export ROS_DISTRO=noetic
export ROS_WORKSPACE=~/agile_ws
export FLIGHTMARE_PATH=$ROS_WORKSPACE/src/agile_flight/flightmare
export PYTHON_ENV_PATH=$(pwd)/flightmare-venv

# --- Exit on error ---
set -e

# --- ROS Setup ---
echo "[INFO] Setting up ROS environment..."
source /opt/ros/$ROS_DISTRO/setup.bash

# --- Create ROS workspace if not exists ---
if [ ! -d "$ROS_WORKSPACE/src" ]; then
  mkdir -p $ROS_WORKSPACE/src
fi

# --- Clone Flightmare if missing ---
if [ ! -d "$FLIGHTMARE_PATH" ]; then
  echo "[INFO] Cloning Flightmare..."
  cd $ROS_WORKSPACE/src
  git clone https://github.com/uzh-rpg/flightmare.git agile_flight
fi

# --- Build ROS workspace ---
cd $ROS_WORKSPACE
catkin_make
source devel/setup.bash
echo "source $ROS_WORKSPACE/devel/setup.bash" >> ~/.bashrc

# --- Python virtual environment ---
echo "[INFO] Creating Python virtualenv at $PYTHON_ENV_PATH..."
cd "$(dirname $PYTHON_ENV_PATH)"
python3 -m venv $(basename $PYTHON_ENV_PATH)
source $PYTHON_ENV_PATH/bin/activate
pip install --upgrade pip

# --- Python dependencies ---
echo "[INFO] Installing Python dependencies..."
pip install -r $(pwd)/requirements.txt

# --- Add project to PYTHONPATH ---
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)" >> ~/.bashrc

# --- Done ---
echo "[✅ SETUP COMPLETE]"
echo "[INFO] Run the following to start training:]"
echo "  source setup.sh && python train.py --episodes 10 --device cpu"
