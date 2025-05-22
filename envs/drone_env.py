# envs/drone_env.py

import gym
from gym import spaces
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import imageio  # NEW: For saving rollouts

from rl.utils import ros_img_to_numpy, compute_goal_reward, log_metrics


class DroneSimEnv(gym.Env):
    def __init__(self, render_eval=False):
        super().__init__()

        rospy.init_node('drone_rl_env', anonymous=True)

        # Action: [vx, vz, yaw_rate]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: image + scalar state
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        })

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/camera/image_raw', Image, self._image_callback)
        rospy.Subscriber('/odom', Odometry, self._odom_callback)
        rospy.wait_for_service('/reset_sim')
        self.reset_sim = rospy.ServiceProxy('/reset_sim', Empty)

        self.latest_image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.current_state = np.zeros(6)
        # TODO: Add random goal position in reset() for generalization
        self.goal_position = np.array([5.0, 0.0, 1.0])
        self.current_step = 0

        # NEW: logging for GIFs
        self.render_eval = render_eval
        self.frame_log = []

    def _image_callback(self, msg):
        self.latest_image = ros_img_to_numpy(msg)

    def _odom_callback(self, msg):
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        self.current_state = np.array([pos.x, pos.y, pos.z, vel.x, vel.y, vel.z])

    def reset(self):
        self.reset_sim()
        rospy.sleep(0.5)
        self.current_step = 0
        self.frame_log = []  # Clear previous frames

        return {
            "image": self.latest_image,
            "state": self.current_state
        }

    def step(self, action):
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.linear.z = float(action[1])
        cmd.angular.z = float(action[2])
        self.cmd_pub.publish(cmd)

        rospy.sleep(0.1)
        self.current_step += 1

        if self.render_eval:
            self.frame_log.append(self.latest_image.copy())

        obs = {
            "image": self.latest_image,
            "state": self.current_state
        }

        reward, reached_goal = compute_goal_reward(self.current_state[:3], self.goal_position)
        done = reached_goal or self.current_step > 200

        log_metrics(self.current_step, reward, self.current_state[:3], self.goal_position, done)

        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    # NEW: Utility function to save a GIF from the recorded frames
    def save_episode_gif(self, filename='rollout.gif', fps=10):
        if self.frame_log:
            imageio.mimsave(filename, self.frame_log, fps=fps)
