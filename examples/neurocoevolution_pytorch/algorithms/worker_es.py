import torch

from algorithms.worker_ea import EAWorker
from utils_torch.chromosome import VBNChromosome
import numpy as np
import torch.nn as nn
PLAYER_1_ID = 'robot_0'
PLAYER_2_ID = 'robot_1'


class ESWorker(EAWorker):
    """ Worker class for the Coevolutionary Evolution Strategies.
    This class handles both the evaluation and mutation of individuals.
    After evaluation, the results are communicated back to the Trainer"""

    def __init__(self, config):
        super().__init__(config)
        self.player1 = VBNChromosome(number_actions=self.config['number_actions'])
        self.player2 = VBNChromosome(number_actions=self.config['number_actions'])
        self.random_player = VBNChromosome(number_actions=self.config['number_actions'])
        self.wall_pass = 0
        self.goal_reach = 0

    def mutate_individual(self, individual):
        """ Mutate the inputted weights and evaluate its performance against the
        inputted oponent. """

        weights = individual
        self.player1.set_weights(weights)
        # noise = torch.randn(1)
        weight_vector = nn.utils.parameters_to_vector(self.player1.model.parameters())
        noise = torch.randn(weight_vector.shape)
        weight_vector = weight_vector + 0.1*noise
        nn.utils.vector_to_parameters(weight_vector, self.player1.model.parameters())
        # for key in weights.keys():
            # noise = np.random.normal(loc=0.0, scale=self.mutation_power, size=weights[key].shape)
            # weights[key] = weights[key] + noise*self.mutation_power
        return (self.player1.get_weights(),noise.detach().cpu().numpy())


    def evaluate_player_fitness(self, player1, player2):
        self.player1.set_weights(player1)
        self.player2.set_weights(player2)
        # reward1 =0
        # reward2 =0
        # for i in range(3):
        reward_1, reward_2, ts, wall_pass, goal_reach = self.play_game(self.player1, self.player2)
        self.wall_pass += wall_pass
        self.goal_reach += goal_reach
        if wall_pass or goal_reach:
            torch.save(self.player1, 'player1_{0}.pth'.format(reward_1))
            torch.save(self.player2, 'player2_{0}.pth'.format(reward_1))
        # reward1 += reward_1
        # reward2 += reward_2
        return (reward_1)


    def evaluate_team_fitness(self, player1, player2):
        self.player1.set_weights(player1)
        self.player2.set_weights(player2)
        # reward1 =0
        # reward2 =0
        # for i in range(3):
        reward_1, reward_2, ts, wall_pass, goal_reach = self.play_team(self.player1, self.player2)
        self.wall_pass += wall_pass
        self.goal_reach += goal_reach
        if wall_pass or goal_reach:
            torch.save(self.player1, 'player1_{0}.pth'.format(reward_1))
            torch.save(self.player2, 'player2_{0}.pth'.format(reward_1))
        # reward1 += reward_1
        # reward2 += reward_2
        return (reward_1 + reward_2)/2