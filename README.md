```markdown
# 🛩️ Ozires: Custom Reinforcement Learning for Autonomous Drone Navigation

**Course Project - CS224R**

![Ozires Logo](ola.png)

**Ozires** is a modular reinforcement learning framework for autonomous drone navigation using vision. It integrates a custom PPO implementation with the DodgeDrone simulation environment (Unity + ROS), supporting both vision-based and state-based policies.

---

## ✨ Features

- 🧠 Custom PPO (Proximal Policy Optimization)
- 🔌 ROS-compatible Gym environment
- 📸 Vision-based and state-based training support
- 🎮 Unity 3D simulation via Flightmare
- 🛫 Designed for sim-to-real deployment

---

## 📁 Project Structure

```
Ozires/
├── rl/                   # PPO logic (agent, actor, critic, buffer, trainer)
├── envs/                 # Custom Gym environment (ROS + Unity)
├── sim/                  # Submodule: DodgeDrone simulator
├── data/                 # Training logs and environment configs
├── models/               # Saved checkpoints
├── gifs/                 # Rollout visualizations
├── train.py              # Training entry point
├── setup.sh              # Environment setup script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## ✅ Prerequisites

- Ubuntu 20.04
- ROS Noetic
- Unity with Flightmare
- Python 3.8
- `virtualenv`

---

## 🔧 Setup

### 1. Clone repository

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/Ozires.git
cd Ozires
```

### 2. Set up Python environment

```bash
python3 -m venv flightmare-venv
source flightmare-venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up ROS environment

If not already done:
```bash
sudo apt update
sudo apt install ros-noetic-desktop-full
```

Then:

```bash
cd ~/agile_ws
catkin_make
source devel/setup.bash
```

### 4. Configure environment

```bash
source setup.sh
```

This sets up:
- ROS paths
- Python virtualenv
- `PYTHONPATH`

---

## 🚀 Launch the simulator

### Without GUI (VM or headless server):

```bash
cd ~/agile_ws
source devel/setup.bash
roslaunch envsim visionenv_sim.launch render:=false gui:=false
```

### With GUI (optional, if you have a display):

```bash
roslaunch envsim visionenv_sim.launch render:=true gui:=true
```

---

## 🧠 Train the agent

From the Ozires project root:

```bash
cd ~/ma/Ozires
source flightmare-venv/bin/activate
source ~/agile_ws/devel/setup.bash
python train.py --episodes 10 --device cpu
```

> Use `--device cuda` if you're running on a GPU.

---

## 🧪 Verify ROS Topics

Run in another terminal:

```bash
source ~/agile_ws/devel/setup.bash
rostopic list
rostopic echo /kingfisher/dodgeros_pilot/state
```

You should see the drone's state info streaming in.

---

## 🧹 Common Issues

- **Stuck at 0,0,0**: Unity may have crashed or `/command` topic is not being published correctly
- **Bridge failed**: Try restarting the Unity simulator and running `roslaunch` again
- **Images not saved**: Check that `cv2` is imported and directories like `debug_images/` exist

---

## 📄 License

MIT License

## 🙏 Acknowledgments

- [DodgeDrone by UZH-RPG](https://github.com/uzh-rpg/agile_flight)
- CS224R Teaching Staff
```
