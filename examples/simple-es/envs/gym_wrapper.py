import gym
import pybullet_envs
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import numpy as np
gym.logger.set_level(40)


def dist_goal_robots(ob):
    goal_position = np.array([8., 0.0, 0.])
    robot_positions = np.zeros((2, 3))
    for i, key in enumerate(ob.keys()):
        # get robot position
        robot_positions[i, :] = ob[key]['joint_state']['position']
    robot_centroid = robot_positions.mean(axis=0)
    return np.linalg.norm(robot_centroid - goal_position), np.linalg.norm(robot_positions[0, :] - robot_positions[1, :])


def flatten_observation(observation_dictonary: dict) -> np.ndarray:
    observation_list = []
    for val in observation_dictonary.values():
        if isinstance(val, np.ndarray):
            observation_list += val.tolist()
        elif isinstance(val, dict):
            observation_list += flatten_observation(val).tolist()
    observation_array = np.array(observation_list)
    return observation_array

def make_env(render=False):
    robots = [
        GenericUrdfReacher(urdf="/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/loadPointRobot.urdf", mode="vel"),
        GenericUrdfReacher(urdf="/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/loadPointRobot.urdf", mode="vel"),
    ]
    env = gym.make("urdf-env-v1", dt=0.1, robots=robots, render=render)
    # Choosing arbitrary actions
    base_pos = np.array(
        [
            [0.0, 0.5, 0.0],
            [0.0, -0.5, 0.0],
        ]
    )
    env.reset(base_pos=base_pos)
    env.add_stuff()
    ob = env.get_observation()
    return env, ob

def kill_env(env):
    env.close()
    del env


class GymWrapper:
    def __init__(self, name, max_step=None, pomdp=False):
        # self.env = gym.make(name)
        print("urdf-env-v1")
        self.env, _ = make_env(render=False)
        if pomdp:
            if "LunarLander" in name:
                print("POMDP LunarLander")
                self.env = LunarLanderPOMDP(self.env)
            elif "CartPole" in name:
                print("POMDP CartPole")
                self.env = CartPolePOMDP(self.env)
            # elif "urdf-env-v1" in name:
            #     print("urdf-env-v1")
            #     self.env,_ = make_env(render=False)
            else:
                raise AssertionError(f"{name} doesn't support POMDP.")
        self.max_step = max_step
        self.curr_step = 0
        self.name = name

    def reset(self):
        self.curr_step = 0
        return_list = {}
        transition = {}
        kill_env(self.env)
        self.env, s = make_env(render=False)
        transition["state"] = flatten_observation(s)
        return_list["0"] = transition
        return return_list

    def step(self, action):
        self.curr_step += 1
        return_list = {}
        transition = {}
        actions = np.clip(np.hstack((action["0"].reshape(2, 2), np.zeros((2, 1)))).ravel(), -0.5, 0.5)
        s, r, d, info = self.env.step(actions)
        s = flatten_observation(s)
        r = r[0]+r[1]

        if self.max_step != "None":
            if self.curr_step >= self.max_step or d:
                d = True
        transition["state"] = s
        transition["reward"] = r
        transition["done"] = d
        transition["info"] = info
        return_list["0"] = transition
        return return_list, r, d, info

    def get_agent_ids(self):
        return ["0"]

    def render(self):
        return self.env.render(mode="rgb_array")

    def close(self):
        kill_env(self.env)


class LunarLanderPOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # modify obs
        obs[2] = 0
        obs[3] = 0
        obs[5] = 0
        return obs


class CartPolePOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # modify obs
        obs[1] = 0
        obs[3] = 0
        return obs
