# Ozires: Custom Reinforcement Learning for Autonomous Drone Navigation

**Project for class: CS224R**


![Logo](ola.png)

**Ozires** is a modular reinforcement learning framework designed to train autonomous drone navigation agents in simulated environments with the goal of real-world deployment. The project integrates a custom PPO (Proximal Policy Optimization) algorithm with a vision-capable drone simulator from the DodgeDrone competition.

## Key Features

- Custom implementation of PPO (Proximal Policy Optimization)
- Modular design with separate components for agent, actor, critic, buffer, and trainer
- Support for both vision-based (camera) and state-based (position, velocity) observations
- Integration with the DodgeDrone simulator (Unity + ROS)
- ROS-compatible Gym environment wrapper for reinforcement learning
- Designed for sim-to-real transfer with real drone deployment in mind

## Project Structure

```
Ozires/
├── rl/                   # Custom RL logic
│   ├── agent.py          # Main agent loop
│   ├── actor.py          # Policy network
│   ├── critic.py         # Value network
│   ├── buffer.py         # Replay buffer and trajectory storage
│   ├── trainer.py        # PPO optimizer
│   └── utils.py          # Helpers (e.g., logging, preprocessing)
│
├── envs/
│   └── drone_env.py      # Custom Gym-compatible drone environment (ROS + Unity)
│
├── deployment/
│   └── deploy_real_drone.py  # Placeholder for real-drone inference using MAVROS
│
├── sim/
│   └── agile_flight/     # Submodule: DodgeDrone simulator (Unity + ROS)
│
├── models/               # Trained models (.pt or .zip)
├── data/                 # Logs, configs, environment maps
│
├── train.py              # Training entry point
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Getting Started

### 1. Clone the repository

```bash
git clone --recurse-submodules git@github.com:guinterface/Ozires.git
cd Ozires
```

### 2. Set up your Python environment

```bash
python -m venv .venv
.venv\Scripts\activate      # On Windows
# source .venv/bin/activate  # On macOS/Linux

pip install -r requirements.txt
```

### 3. Launch the Unity simulator

```bash
cd sim/agile_flight
./setup_ros.bash            # Or setup_py.bash if using Python interface
roslaunch envsim visionenv_sim.launch render:=True
```

### 4. Train your model

```bash
python train.py
```

## Simulator Integration

The project uses the **DodgeDrone** simulator developed at UZH-RPG, providing a photorealistic Unity-based environment and ROS interface. This setup allows realistic physics and obstacle-rich training environments for vision or state-based navigation.

## Future Work

- Curriculum learning for complex navigation tasks
- Domain randomization for robust sim-to-real transfer
- Real drone deployment via MAVROS (PX4 or ArduPilot)
- Integration of wind zones, dynamic obstacles, and 3D perception

## License

This project is open-source and distributed under the MIT License.

## Acknowledgments

- [DodgeDrone: UZH-RPG Drone Racing Simulator](https://github.com/uzh-rpg/agile_flight)
