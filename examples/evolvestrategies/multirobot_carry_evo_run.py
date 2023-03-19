import tqdm as tqdm
from torch.multiprocessing import Pool
from torch.optim import Adam

from evostrat import compute_centered_ranks, NormalPopulation
from MRCarry import MRCarry
import wandb
import pickle
import torch
from typing import Dict
import gym
from torch import nn
import torch as t
from evostrat import Individual
import numpy as np

from urdfenvs.robots.generic_urdf import GenericUrdfReacher


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
            [0.0, .75, 0.0],
            [0.0, -.75, 0.0],
        ]
    )
    env.reset(base_pos=base_pos)
    env.add_stuff()
    ob = env.get_observation()
    return env, ob

def kill_env(env):
    env.close()
    del env


# state_params = pickle.load(open('/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/Evolutionary-Reinforcement-Learning/simple_no_pool/Models/3action_best.pt', 'rb'))


mrc1 = MRCarry()
# mrc1.net.load_state_dict(state_params)
mrc1.actor1.load_state_dict(torch.load('/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/Evolutionary-Reinforcement-Learning/simple_no_pool/Models/1_actor_800.0.pt'), strict=False)
import numpy as np
import pybullet as p

for i in range(100):
    env, obs = make_env(render=True)
    # p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/evolvestrategies/evo3actions.mp4")
    obs = flatten_observation(obs)
    done = False
    r_tot = 0
    alpha = 0.5
    action_prev = torch.zeros((6,))
    rf = 0

    while not done:
        action = mrc1.actor1(torch.FloatTensor(obs))
        # actions = np.clip(np.hstack((action.detach().numpy().reshape(2, 2), np.zeros((2, 1)))).ravel(), -0.5, 0.5)
        # action_new = alpha * action + (1 - alpha) * action_prev
        # action_prev = action_new
        obs, rew, done, info = env.step(action.ravel())
        r_tot += (rew[0] + rew[1]) * 0.5 + rf * 10
        obs = flatten_observation(obs)
        print(rew[0] + rew[1])
        if info['goal_reached']:
            print("Goal Reached")
            break
    kill_env(env)

# p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)

