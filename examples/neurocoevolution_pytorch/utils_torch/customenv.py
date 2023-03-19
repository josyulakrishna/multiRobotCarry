import gym
import numpy as np

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import pandas as pd

#create datframe with collumns distance and penalty
df = pd.DataFrame(columns=['distance', 'penalty'])
def flatten_observation(observation_dictonary: dict) -> np.ndarray:
    observation_list = []
    for val in observation_dictonary.values():
        if isinstance(val, np.ndarray):
            observation_list += val.tolist()
        elif isinstance(val, dict):
            observation_list += flatten_observation(val).tolist()
    observation_array = np.array(observation_list)
    return observation_array

def get_bc_features(ob):
    goal_position = np.array([8., 0.0, 0.])
    robot_positions = np.zeros((2, 3))
    for i, key in enumerate(ob.keys()):
        #get robot position
        robot_positions[i,:] = ob[key]['joint_state']['position']
    robot_centroid = robot_positions.mean(axis=0)
    slope_goal = (goal_position[1] - robot_centroid[1])/(goal_position[0] - robot_centroid[0])
    slope_robots = (robot_positions[1][1] - robot_positions[0][1])/((robot_positions[1][0] - robot_positions[0][0])+0.001)
    return np.linalg.norm(robot_centroid - goal_position), np.linalg.norm(robot_positions[0,:] - robot_positions[1,:] ), slope_goal, slope_robots



def make_env(render=False):
    robots = [
        GenericUrdfReacher(urdf="/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/loadPointRobot.urdf", mode="vel"),
        GenericUrdfReacher(urdf="/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/loadPointRobot.urdf", mode="vel"),
    ]
    env = gym.make("urdf-env-v1", dt=0.1, robots=robots, render=render)
    # Choosing arbitrary actions
    base_pos = np.array(
        [
            [0.0, 0.75, 0.0],
            [0.0, -0.75, 0.0],
        ]
    )
    env.reset(base_pos=base_pos)
    env.add_stuff()
    ob = env.get_observation()
    return env, ob

def kill_env(env):
    env.close()
    del env
