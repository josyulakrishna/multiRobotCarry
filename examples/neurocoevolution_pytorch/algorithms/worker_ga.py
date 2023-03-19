
from algorithms.worker_ea import EAWorker
from utils_torch.chromosome import VBNChromosome
import numpy as np


PLAYER_1_ID = 'robot_0'
PLAYER_2_ID = 'robot_1'

class GAWorker(EAWorker):
    """ Worker class for the Coevolutionary Genetic Algorithm.
    This class handles both the evaluation and mutation of individuals.
    After evaluation, the results are communicated back to the Trainer"""

    def __init__(self, config):
        super().__init__(config)

        self.player1 = VBNChromosome(number_actions=self.config['number_actions'])
        self.player2 = VBNChromosome(number_actions=self.config['number_actions'])
        self.random_player = VBNChromosome(number_actions=self.config['number_actions'])

        self.wall_pass = 0
        self.goal_reach = 0
        self.mutation_power = self.config['mutation_power']
    #mutate individual get weights
    def mutate_individual(self, individual):
        """ Mutate the inputted weights and evaluate its performance against the
        inputted oponent. """

        weights = individual
        for key in weights.keys():
            noise = np.random.normal(loc=0.0, scale=self.mutation_power, size=weights[key].shape)
            weights[key] = weights[key] + noise
        return weights

    def evaluate_team_fitness(self, player1, player2):
        self.player1.set_weights(player1)
        self.player2.set_weights(player2)
        reward1 =0
        reward2 =0
        for i in range(3):
            reward_1, reward_2, ts, wall_pass, goal_reach = self.play_game(self.player1, self.player2)
            self.wall_pass += wall_pass
            self.goal_reach += goal_reach
            reward1 += reward_1
            reward2 += reward_2
        return (reward1)/3.

    def evaluate_hof_diff(self, player1, hofplayer):
        """ Mutate the inputted weights and evaluate its performance against the
        inputted oponent. """
        # recorder = VideoRecorder(self.env, path=self.video_path) if record else None
        self.player1.set_weights(player1)
        self.player2.set_weights(hofplayer)
        self.random_player.mutate_weights()

        elite_reward1, oponent_reward1, ts1, wall_pass, goal_reach = self.play_game(self.player1, self.player2)

        elite_reward2,oponent_reward2, ts2, wall_pass, goal_reach = self.play_game(self.player1, self.random_player)
        total_elite = elite_reward1 + oponent_reward1
        total_oponent = elite_reward2 + oponent_reward2
        if total_elite <0 and  total_oponent<0:
            return -1*abs(total_elite - total_oponent)
        else:
            return total_elite - total_oponent
