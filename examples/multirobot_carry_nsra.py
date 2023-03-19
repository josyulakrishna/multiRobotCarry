import collections

import gym
import torch
import numpy as np
from estorch import NSRA_ES
from estorch import ES
import ipdb
import logging
import wandb
wandb.init(project="nsra", entity="josyula")
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

def normal(triangles):
    # The cross product of two sides is a normal vector
    return np.cross(triangles[:,1] - triangles[:,0],
                    triangles[:,2] - triangles[:,0], axis=1)

def area(triangles):
    # The norm of the cross product of two sides is twice the area
    return np.linalg.norm(normal(triangles), axis=1) / 2

class Agent():
    """NS-ES, NSR-ES, NSRA-ES algorithms require additional signal in addition to the
    reward signal. This signal is called behaviour characteristics and it is domain
    dependent signal which has to be chosen by the user. For more information look
    into references."""
    def __init__(self, device=torch.device('cpu'), n=32):
        self.env = None #, self.ob_init = make_env(render=True)
        self.device = device

        # Number of features to be used as behaviour characteristics
        # distance from goal, distance between robots, slope of goal, slope of robots
        # n is the behaviour charecteristic number.
        self.n = n

    def rollout(self, policy, render=False):
        done = False
        # kill_env(self.env)
        if self.env:
            self.env.close()
        self.env, observation = make_env(render=render)
        step = 0
        total_reward = 0
        start_actions = collections.deque(maxlen=self.n)
        last_actions = collections.deque(maxlen=self.n)
        action_prev = np.zeros((6,))
        alpha = 0.5
        robot_positions = np.zeros((2, 3))
        goal_pos = np.array([8.0, 0, 0])
        tri_areas, goal_dist = [], []
        with torch.no_grad():
            while not done:
                observation = (torch.from_numpy(flatten_observation(observation))
                               .float()
                               .to(self.device))
                action = (policy(observation)
                        .data
                        .detach()
                        .cpu()
                        .numpy())
                actions = np.clip(np.hstack((action.reshape(2, 2), np.zeros((2, 1)))).ravel(), -0.5, 0.5)
                action_new = alpha * actions + (1 - alpha) * action_prev
                # action_new = np.clip(action_new, -0.1, 0.1)
                observation, reward, done, info = self.env.step(action_new)
                action_prev = action_new
                dist2goal, distbrobots, slopegoal, sloperobots = get_bc_features(observation)
                for i, key in enumerate(observation.keys()):
                    # get robot position
                    robot_positions[i, :] = observation[key]['joint_state']['position']
                area_tri = abs(np.cross(robot_positions[0, :] - robot_positions[1, :], goal_pos - robot_positions[0, :]))
                area_tri = np.linalg.norm(area_tri) / 2
                step += 1
                penalty = -(area_tri+0.001)*np.exp(-(dist2goal/8)+0.001)
                total_reward += (reward[0]+reward[1])+penalty*10
                if total_reward>0:
                    pass
                last_actions.append(robot_positions[:,:2].ravel())
                # last_actions.append(np.array([dist2goal, distbrobots, slopegoal, sloperobots]))
                if step<self.n:
                    start_actions.append(robot_positions[:,:2].ravel())
                    # start_actions.append(np.array([dist2goal, distbrobots, slopegoal, sloperobots]))
                if step > 3000: break
                step+=1
        bc = np.concatenate(last_actions).flatten().reshape(-1, 4)
        return total_reward, bc

class Policy(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(Policy, self).__init__()
        self.linear_1 = torch.nn.Linear(n_input, 64)
        self.activation_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(64, 64)
        self.activation_2 = torch.nn.ReLU()
        self.linear_3 = torch.nn.Linear(64, n_output)


    def forward(self, x):
        l1 = self.linear_1(x)
        a1 = self.activation_1(l1)
        l2 = self.linear_2(a1)
        a2 = self.activation_2(l2)
        l3 = torch.nn.functional.tanh(self.linear_3(a2))
        return l3


if __name__ == '__main__':
    # Usages for other novelty search algorithms are the same.
    # NS_ES ignores reward signals and optimizes purely for the novelty
    # NSR_ES uses novelty and reward signals equally.
    # NSRA_ES adaptively changes the importance of the signals.
    device = torch.device("cuda:0")
    agent = Agent(n=1)
    n_input = 28 #agent.env.observation_space.shape[0]
    n_output = 4 #agent.env.action_space.shape[0]
    es = NSRA_ES(Policy, Agent, torch.optim.Adam, population_size=10, sigma=0.1,
                 weight_t= 10,
                 device=device, policy_kwargs={'n_input': n_input, 'n_output': n_output},
                 agent_kwargs={'device': device}, optimizer_kwargs={'lr': 0.1})

    es.train(n_steps=200, n_proc=1)

    # Meta Population Rewards
    for idx, (policy, _) in enumerate(es.meta_population):
        reward, bc = agent.rollout(policy, render=True)
        print(f'Reward of {idx}. policy from the meta population: {reward}')
        wandb.log({"Reward": reward, "idx": idx})

    # Policy with the highest reward
    policy = Policy(n_input, n_output).to(device)
    policy.load_state_dict(es.best_policy_dict)
    torch.save(policy.state_dict(), 'best_policy.pth')
    reward, bc = agent.rollout(policy, render=True)
    print(f'Best Policy Reward: {reward}')