#!/bin/bash

set -e  # Exit on error

echo "🚀 Setting up Ozires environment on Ubuntu (ROS Noetic + PPO + Unity sim)..."

# 1. ROS Noetic Installation (Ubuntu 20.04)
echo "🔧 Installing ROS Noetic..."
sudo apt update && sudo apt upgrade -y

# Add ROS repo
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl -y
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update

# Install full desktop ROS
sudo apt install -y ros-noetic-desktop-full

# ROS environment setup
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source /opt/ros/noetic/setup.bash

# ROS Python tools
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential python3-catkin-tools

# Initialize rosdep
sudo rosdep init || true
rosdep update

# 2. Python dependencies
echo "🐍 Installing Python dependencies..."
sudo apt install -y python3-pip python3-opencv
pip3 install --upgrade pip

# 3. Set up your repo
echo "📦 Cloning your repo..."
cd ~
git clone --recurse-submodules git@github.com:guinterface/Ozires.git
cd Ozires/sim/agile_flight

# 4. Build ROS workspace
echo "🛠️ Building simulator..."
./setup_ros.bash

# 5. Install project Python packages
cd ~/Ozires
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "To run the simulator:"
echo "  cd ~/Ozires/sim/agile_flight"
echo "  source devel/setup.bash"
echo "  roslaunch envsim visionenv_sim.launch render:=True"
