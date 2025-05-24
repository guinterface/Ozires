#!/bin/bash
set -e

echo ">>> Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo ">>> Installing Python dependencies..."
pip install torch==2.4.0 torchvision gym==0.26.2 numpy opencv-python imageio tensorboard rospkg catkin_pkg PyYAML tqdm matplotlib

echo ">>> Setting up ROS (Ubuntu only)..."
sudo apt update
sudo apt install -y ros-noetic-desktop-full python3-catkin-tools python3-opencv ros-noetic-cv-bridge
sudo rosdep init || true
rosdep update

echo ">>> Creating catkin workspace..."
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin build
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source devel/setup.bash

echo ">>> Setup complete! ✅"
