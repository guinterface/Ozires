# envs/drone_env.py

import gym
from gym import spaces
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Imu  # TODO: Add IMU processing
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

class DroneSimEnv(gym.Env):
    def __init__(self):
        super().__init__()

        rospy.init_node('drone_rl_env', anonymous=True)

        # Action: [vx, vz, yaw_rate]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # TODO: Add LIDAR or depth image if available
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # TODO: Expand with IMU/goal
        })

        # Publishers and Subscribers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/camera/image_raw', Image, self._image_callback)
        rospy.Subscriber('/odom', Odometry, self._odom_callback)

        # TODO: Subscribe to IMU if needed
        # rospy.Subscriber('/imu', Imu, self._imu_callback)

        # Reset service
        rospy.wait_for_service('/reset_sim')
        self.reset_sim = rospy.ServiceProxy('/reset_sim', Empty)

        self.latest_image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.current_state = np.zeros(6)  # TODO: Consider [px, py, pz, vx, vy, vz, yaw, goal_dx, goal_dy, ...]

        # TODO: Track goal position and calculate distance to goal
        # self.goal_position = np.array([10.0, 0.0, 1.0])

    def _image_callback(self, msg):
        # TODO: Use cv_bridge to convert ROS Image to NumPy array
        self.latest_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    def _odom_callback(self, msg):
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        # TODO: Add orientation (e.g., yaw)
        self.current_state = np.array([pos.x, pos.y, pos.z, vel.x, vel.y, vel.z])

    # def _imu_callback(self, msg):
    #     # TODO: Extract orientation, acceleration, angular velocity if needed
    #     pass

    def reset(self):
        self.reset_sim()
        rospy.sleep(0.5)

        obs = {
            "image": self.latest_image,
            "state": self.current_state  # TODO: Add goal-relative vector if available
        }
        return obs

    def step(self, action):
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.linear.z = float(action[1])
        cmd.angular.z = float(action[2])
        self.cmd_pub.publish(cmd)

        rospy.sleep(0.1)

        obs = {
            "image": self.latest_image,
            "state": self.current_state
        }

        reward = -1.0  # TODO: Define meaningful reward (e.g., -distance, +goal, -collision)
        done = False   # TODO: Define termination (goal reached, crash, timeout)

        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
