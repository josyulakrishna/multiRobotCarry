from abc import abstractmethod, ABCMeta

import numpy as np
from customenv import *
import torch
PLAYER_1_ID = "robot_0"
PLAYER_2_ID = "robot_1"




class EAWorker:
    """ Class that includes some functionality that is used by both the
    Evolution Strategies and Genetic Algorithm Workers. """

    __metaclass__ = ABCMeta

    def __init__(self,
                 config):

        self.config = config
        self.mutation_power = config['mutation_power']

    def evaluate(self, weights):
        """ Evlauate weights by playing against a random policy. """
        # recorder = VideoRecorder(self.env, path=self.video_path_eval)
        self.elite.set_weights(weights)
        reward, _, ts = self.play_game(self.elite,
                                       None,
                                       recorder=None,
                                       eval=True)
        return {
            'total_reward': reward,
            'timesteps_total': ts
        }

    def play_game(self, player1, player2):
        """ Play a game using the weights of two players. """
        env, obs = make_env(render=False)
        reward1 = 0
        reward2 = 0
        limit = self.config['max_evaluation_steps'] if eval else self.config[
            'max_timesteps_per_episode']
        step = 0
        wall_pass = 0
        goal_reach = 0
        action_prev = np.zeros((6,))
        alpha = 0.9
        for ts in range(limit):
            filtered_obs1 = flatten_observation(obs[PLAYER_1_ID])
            filtered_obs2 = flatten_observation(obs[PLAYER_2_ID])
            filtered_obs1 = torch.FloatTensor(filtered_obs1)  #.to(self.device)
            filtered_obs2 = torch.FloatTensor(filtered_obs2)
            action1 = torch.FloatTensor(np.random.randn(3,)) if player1==None else player1.determine_actions(filtered_obs1)
            action2 = torch.FloatTensor(np.random.randn(3,)) if player2==None else player2.determine_actions(filtered_obs2)
            actions = np.stack([action1.detach().cpu().numpy(), action2.detach().cpu().numpy()]).ravel()
            actions = alpha*action_prev + (1-alpha) * (actions)
            action_prev = actions
            actions = np.clip(actions, -1, 1)
            # actions = np.hstack((actions.reshape(2, 2), np.zeros((2, 1)))).ravel()
            obs, reward, done, info = env.step(actions)
            # dist2goal, distbrobots, slopegoal, sloperobots = get_bc_features(obs)
            # if dist2goal > 0.2:
            #     rf = 1/(dist2goal)
            # fitness += r[0] + r[1]+rf
            reward1 += reward[0]
            reward2 += reward[1]
            step+= 1
            if info["wall_pass"]:
                wall_pass += 1
            if info["goal_reached"]:
                goal_reach += 1
            if done or step >= limit:
                break
        kill_env(env)
        return reward1, reward2, ts, wall_pass, goal_reach


    def play_team(self, player1, player2):
        """ Play a game using the weights of two players. """
        env, obs = make_env(render=False)
        reward1 = 0
        reward2 = 0
        limit = self.config['max_evaluation_steps'] if eval else self.config[
            'max_timesteps_per_episode']
        step = 0
        wall_pass = 0
        goal_reach = 0
        action_prev = np.zeros((6,))
        alpha = 0.9
        for ts in range(limit):
            filtered_obs1 = flatten_observation(obs[PLAYER_1_ID])
            filtered_obs2 = flatten_observation(obs[PLAYER_2_ID])
            filtered_obs1 = torch.FloatTensor(filtered_obs1)  #.to(self.device)
            filtered_obs2 = torch.FloatTensor(filtered_obs2)
            action1 = torch.FloatTensor(np.random.randn(3,)) if player1==None else player1.determine_actions(filtered_obs1)
            action2 = torch.FloatTensor(np.random.randn(3,)) if player2==None else player2.determine_actions(filtered_obs2)
            actions = np.stack([action1.detach().cpu().numpy(), action2.detach().cpu().numpy()]).ravel()
            # actions = alpha*action_prev + (1-alpha) * (actions)
            # action_prev = actions
            actions = np.clip(actions, -0.5, 0.5)
            # actions = np.hstack((actions.reshape(2, 2), np.zeros((2, 1)))).ravel()
            obs, reward, done, info = env.step(actions)
            # dist2goal, distbrobots, slopegoal, sloperobots = get_bc_features(obs)
            # if dist2goal > 0.2:
            #     rf = 1/(dist2goal)
            # fitness += r[0] + r[1]+rf
            reward1 += reward[0]
            reward2 += reward[1]
            step+= 1
            if info["wall_pass"]:
                wall_pass += 1
            if info["goal_reached"]:
                goal_reach += 1
            if done or step >= limit:
                break
        kill_env(env)
        return reward1, reward2, ts, wall_pass, goal_reach
