# rl/utils.py

import numpy as np
import math
import cv2

from cv_bridge import CvBridge
bridge = CvBridge()



# =========================
# IMAGE UTILITIES
# =========================

def ros_img_to_numpy(msg, resize_shape=(224, 224)):
    """Convert ROS Image message to a resized NumPy RGB image"""
    try:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = cv2.resize(img, resize_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"[utils] Failed to convert image: {e}")
        return np.zeros((resize_shape[0], resize_shape[1], 3), dtype=np.uint8)

# =========================
# STATE / REWARD HELPERS
# =========================

def compute_distance(a, b):
    """Euclidean distance between two 3D vectors"""
    return np.linalg.norm(np.array(a) - np.array(b))

# TODO: Customize reward function
def compute_goal_reward(position, goal, threshold=0.5):
    """Simple reward: +100 if within threshold of goal, -distance otherwise"""
    dist = compute_distance(position, goal)
    if dist < threshold:
        return 100.0, True
    return -dist, False

# =========================
# NORMALIZATION / CLIPPING
# =========================

def normalize_angle_rad(angle):
    """Normalize angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

# =========================
# METRICS / LOGGING
# =========================

def log_metrics(step, reward, position, goal, success):
    print(f"[Step {step}] Reward: {reward:.2f}, Pos: {position}, Goal: {goal}, Success: {success}")

def buffer_to_tensors(buffer, device):
    images = torch.tensor(buffer.images / 255.0, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    states = torch.tensor(buffer.states, dtype=torch.float32).to(device)
    actions = torch.tensor(buffer.actions, dtype=torch.float32).to(device)
    log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32).to(device)
    returns = torch.tensor(buffer.returns, dtype=torch.float32).to(device)
    advantages = torch.tensor(buffer.advantages, dtype=torch.float32).to(device)
    return images, states, actions, log_probs, returns, advantages

