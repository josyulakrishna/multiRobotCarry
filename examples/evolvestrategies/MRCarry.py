from typing import Dict
import gym
from torch import nn
import torch as t
from evostrat import Individual
import numpy as np

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import ipdb
import logging
import pandas as pd
import wandb

# wandb.init(project="evo_multirobot_carry_local_passage", entity="josyula")

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

class MRCarry(Individual):
    """
    A lunar lander control 3]
b = [10, 25, 17, 11, 13, 17, 20, 13, 9, 15]
from scipy.stats import linregress
linregress(a, b)
will return you:

LinregressResult(slope=0.20833333333333337, intercept=13.375, rvalue=0.14499815458068521, pvalue=0.68940144811669501, stderr=0.50261704627083648)led by a feedforward policy network
    """

    def __init__(self):
        # self.net = nn.Sequential(
        #         nn.Linear(28, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 6),
        #         # nn.ReLU(),
        #         # nn.Linear(64, 4),
        #         nn.Tanh()
        #     )
        self.actor1 = nn.Sequential(
            nn.Linear(28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            # nn.Tanh()
        )
    @staticmethod
    def from_params(params: Dict[str, t.Tensor]) -> 'MRCarry':
        agent = MRCarry()
        agent.net.load_state_dict(params)
        return agent

    def fitness(self, render=False) -> float:
        env, obs = make_env(render=render)
        global df
        obs = flatten_observation(obs)
        done = False
        r_tot = 0
        alpha = 0.5
        action_prev = np.zeros((6,))
        rf=0
        df1 = pd.DataFrame(columns=['distance', 'penalty'])
        db_robots = []
        penalty_ = []
        while not done:
            action = self.action(obs)
            actions = np.clip(np.hstack((action.reshape(2,2), np.zeros((2,1)))).ravel(), -0.5, 0.5)
            action_new = alpha * actions + (1 - alpha) * action_prev
            # action_new = np.clip(action_new, -0.5, 0.5)
            action_prev = action_new
            obs, rew, done, info = env.step(action_new)
            dist_goal, dist_robots, penalty = self.dist_goal_robots(obs)
            # ipdb.set_trace()
            logging.info("Distance robots: {0}, {1}".format(dist_robots,penalty))
            # if dist_goal>=0.2:
            #     rf = 1/dist_goal
            # if dist_robots<=0.2 or dist_robots>=1.5:
            #     penalty = -50
            obs = flatten_observation(obs)
            db_robots.append(dist_robots)
            penalty_.append(penalty)
            r_tot += (rew[0]+rew[1])*0.5+rf*10+penalty
            if info['goal_reached']:
                pass
        wandb_list = zip(db_robots, penalty_)
        df1 = pd.DataFrame(wandb_list, columns=['distance', 'penalty'])
        df = df.append(df1)
        table = wandb.Table(dataframe=df)
        wandb.log("line_plot", wandb.plot.line(table, "distance", "penalty"))
        kill_env(env)
        return r_tot

    def dist_goal_robots(self, ob):

        goal_position = np.array([8., 0.0, 0.])
        robot_positions = np.zeros((2, 3))
        for i, key in enumerate(ob.keys()):
            #get robot position
            robot_positions[i,:] = ob[key]['joint_state']['position']
        robot_centroid = robot_positions.mean(axis=0)
        goal_pos = np.array([8., 0.0, 0.])
        area_tri = abs(np.cross(robot_positions[0, :] - robot_positions[1, :], goal_pos - robot_positions[0, :]))
        area_tri = np.linalg.norm(area_tri) / 2
        dist2goal = np.linalg.norm(robot_centroid - goal_position)
        penalty = -(area_tri + 0.001) * np.exp(-(dist2goal / 8) + 0.001)
        distbetweenrobots =  np.linalg.norm(robot_positions[0,:] - robot_positions[1,:] )
        return  dist2goal, distbetweenrobots, penalty

    def get_params(self) -> Dict[str, t.Tensor]:
        return self.net.state_dict()

    def action(self, obs):
        with t.no_grad():
            return self.net(t.tensor(obs, dtype=t.float32))
