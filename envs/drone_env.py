# envs/drone_env.py

import gym
from gym import spaces
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_srvs.srv import Empty
import imageio
from rl.utils import ros_img_to_numpy, log_metrics



class DroneSimEnv(gym.Env):
    def __init__(self, render_eval=False, use_multimodal=False):
        super().__init__()
        rospy.init_node('drone_rl_env', anonymous=True)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        })

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/camera/image_raw', Image, self._image_callback)
        rospy.Subscriber('/odom', Odometry, self._odom_callback)
        rospy.Subscriber('/imu', Imu, self._imu_callback)
        rospy.wait_for_service('/reset_sim')
        self.reset_sim = rospy.ServiceProxy('/reset_sim', Empty)

        self.latest_image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.current_state = np.zeros(6)
        self.latest_imu = np.zeros(6)
        self.goal_position = np.zeros(3)
        self.prev_distance = None
        self.initial_distance = None
        self.current_step = 0
        self.max_steps = 200

        self.render_eval = render_eval
        self.frame_log = []
        self.use_multimodal = use_multimodal

    def reset(self):
        self.reset_sim()
        rospy.sleep(0.5)
        self.current_step = 0
        self.frame_log = []

        self.goal_position = np.array([
            np.random.uniform(3.0, 8.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(1.0, 3.0)
        ])

        # Geo-fence around the goal
        box_size = np.array([2.0, 2.0, 2.0])
        self.fence_min = self.goal_position - box_size
        self.fence_max = self.goal_position + box_size

        self.prev_distance = np.linalg.norm(self.current_state[:3] - self.goal_position)
        self.initial_distance = self.prev_distance

        print(f"[ENV RESET] New goal: {self.goal_position}")

        return {
            "image": self.latest_image,
            "state": np.concatenate([self.current_state, self.latest_imu])
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
            "state": np.concatenate([self.current_state, self.latest_imu])
        }

        if self.use_multimodal:
            reward, done = self._compute_multimodal_reward()
        else:
            reward, done = self._compute_simple_reward()

        log_metrics(self.current_step, reward, self.current_state[:3], self.goal_position, done)
        return obs, reward, done, {}

    def _compute_simple_reward(self):
        curr_pos = self.current_state[:3]
        dist = np.linalg.norm(curr_pos - self.goal_position)
        delta_dist = self.prev_distance - dist
        self.prev_distance = dist

        reward = -1 + delta_dist
        done = False

        if not np.all((self.fence_min <= curr_pos) & (curr_pos <= self.fence_max)):
            reward = -100
            done = True
            return reward, done

        if dist < 0.5:
            reward = 100
            done = True
            return reward, done

        if self.current_step >= self.max_steps:
            reward = -100
            done = True

        return reward, done

    def _compute_multimodal_reward(self):
        dist = np.linalg.norm(self.current_state[:3] - self.goal_position)
        delta_dist = self.prev_distance - dist
        self.prev_distance = dist

        r_distance = (delta_dist / (self.initial_distance + 1e-6)) * 100
        r_goal = 10 if dist < 0.5 else 0
        r_time = -0.001
        r_timeout = -10 if self.current_step >= self.max_steps else 0
        r_collision = 0  # TODO: Add collision topic check

        reward = r_distance + r_goal + r_time + r_timeout + r_collision
        done = dist < 0.5 or self.current_step >= self.max_steps
        return reward, done

    def _image_callback(self, msg):
        self.latest_image = ros_img_to_numpy(msg)

    def _odom_callback(self, msg):
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        self.current_state = np.array([pos.x, pos.y, pos.z, vel.x, vel.y, vel.z])

    def _imu_callback(self, msg):
        acc = msg.linear_acceleration
        ang = msg.angular_velocity
        self.latest_imu = np.array([
            acc.x, acc.y, acc.z,
            ang.x, ang.y, ang.z
        ])

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def save_episode_gif(self, filename='rollout.gif', fps=10):
        if self.frame_log:
            imageio.mimsave(filename, self.frame_log, fps=fps)
