# import wandb
import numpy as np
import yaml
from collections import deque
from algorithms.trainer_ea import EATrainer
from utils_torch .chromosome import VBNChromosome
from torch.multiprocessing import Pool
import torch
import copy

DEFAULT_CONFIG = {}
with open('configs/config_ga_test.yaml') as f:
    DEFAULT_CONFIG = yaml.load(f, Loader=yaml.FullLoader)


class ESTrainer(EATrainer):
    """ Trainer class for the Coevolutionary Evolution Strategies algorithm.
    This class distributes the mutation and evaluation workload over a number
    of workers and updates the network weights."""

    _name = "ES"
    _default_config = DEFAULT_CONFIG

    def __init__(self, config):

        super(ESTrainer, self).__init__(config)
        self.n = self.config['population_pool']
        self.k = self.config['population_size']
        chromosome = VBNChromosome(number_actions=self.config['number_actions'])
        self.population_player1 = [
             chromosome.get_weights()
        ]
        chromosome1 = VBNChromosome(number_actions=self.config['number_actions'])

        self.population_player2 = [
            chromosome1.get_weights()
        ]
        del chromosome, chromosome1
        self.player1_hof = deque(maxlen=self.k)
        self.player2_hof = deque(maxlen=self.k)
        self.prev_fitness = deque(maxlen=10)
        self.player1_hof_best = deque(maxlen=1)
        self.player2_hof_best = deque(maxlen=1)
        self.rest_mutation_rate = False
        self.summary = {"prev_gen_fit_1": -1000,
                        "prev_gen_fit_2": -1000}

        self.player1_hof.append(self.population_player1[0])
        self.player2_hof.append(self.population_player2[0])
        self.player1_hof_best.append(self.population_player1[0])
        self.player2_hof_best.append(self.population_player2[0])


    def step(self):
        """ Evolve one generation using the Evolution Strategies algorithm.
        This consists of four steps:
        1. Send the current weights to a number of workers and mutate and evaluate them.
        2. Communicate the mutated weights and their fitness back to the Trainer.
        3. Update the weights using the ES update rule.
        4. Evaluate the updated weights against a random policy and log the outcome.
        """
        mutation_pool = Pool(processes=self.n)
        mutations_player1 = []
        mutations_player2 = []
        # self.worker_class.mutate_individual(self.player1_hof[0])
        for i in range(len(self.player1_hof)):
            temp = [copy.deepcopy(self.player1_hof[i])] * self.n
            mutations_player1 += [mutation_pool.map(self.worker_class.mutate_individual, temp)]
        for i in range(len(self.player2_hof)):
            temp = [copy.deepcopy(self.player2_hof[i])] * self.n
            mutations_player2 += [mutation_pool.map(self.worker_class.mutate_individual, temp)]
        mutation_pool.close()
        weights_mutations1 = []
        weights_mutations2 = []
        epsilons_mutations1 = []
        epsilons_mutations2 = []
        for i in range(len(mutations_player1)):
            wm, em = zip(*mutations_player1[i])
            weights_mutations1 += [wm]
            epsilons_mutations1 += [em]
        for i in range(len(mutations_player2)):
            wm, em = zip(*mutations_player2[i])
            weights_mutations2 += [wm]
            epsilons_mutations2 += [em]
        # self.worker_class.evaluate_team_fitness(weights_mutations1[0][2], self.player2_hof_best[0])
        fitness_pool = Pool(processes=self.n)
        #player1 fitness evaluation
        fitness_player1 = []
        for i in range(len(weights_mutations1)):
            fitness_player1 += [fitness_pool.starmap(self.worker_class.evaluate_player_fitness, zip(weights_mutations1[i], [self.player2_hof_best[0]]*self.n))]

        #player2 fitness evaluation
        fitness_player2 = []
        for i in range(len(weights_mutations2)):
            fitness_player2 += [fitness_pool.starmap(self.worker_class.evaluate_player_fitness, zip( [self.player1_hof_best[0]]*self.n, weights_mutations2[i]))]

        fitness_pool.close()
        alpha = 0.0001
        sigma = 0.1
        #update weights of player1
        fitness_player1 = np.array(fitness_player1)
        epsilons_mutations1 = np.array(epsilons_mutations1)


        # total_fitness_player1 = np.sum(np.multiply(fitness_player1, epsilons_mutations1), axis=1)/(len(fitness_player1)*len(fitness_player1[0]))
        # f1 = torch.FloatTensor([(alpha / sigma) * np.sum(total_fitness_player1)])
        temp_player = VBNChromosome(number_actions=self.config['number_actions'])
        temp_player.set_weights(self.population_player1[0])
        total_fitness_player1 = 0
        for i in range(len(fitness_player1)):
            total_fitness_player1 += np.multiply(fitness_player1[i],epsilons_mutations1[i].transpose()).transpose().sum()
        total_fitness_player1 = total_fitness_player1/(len(fitness_player1)*len(fitness_player1[0]))
        weight_vector = torch.nn.utils.parameters_to_vector(temp_player.model.parameters())
        weight_vector = weight_vector + (alpha / sigma) * total_fitness_player1
        torch.nn.utils.vector_to_parameters(weight_vector, temp_player.model.parameters())
        self.population_player1[0] = temp_player.get_weights()
        del temp_player
        #update weights of player2

        fitness_player2 = np.array(fitness_player2)
        epsilons_mutations2 = np.array(epsilons_mutations2)
        # total_fitness_player2 = np.sum(np.multiply(fitness_player2, epsilons_mutations2), axis=1)/(len(fitness_player2)*len(fitness_player2[0]))
        # f2 = torch.FloatTensor([(alpha / sigma) * np.sum(total_fitness_player2)])
        temp_player = VBNChromosome(number_actions=self.config['number_actions'])
        temp_player.set_weights(self.population_player2[0])
        total_fitness_player2 = 0
        for i in range(len(fitness_player2)):
            total_fitness_player2 += np.multiply(fitness_player2[i],epsilons_mutations2[i].transpose()).transpose().sum()
        total_fitness_player2 = total_fitness_player2/(len(fitness_player2)*len(fitness_player2[0]))
        weight_vector = torch.nn.utils.parameters_to_vector(temp_player.model.parameters())
        weight_vector = weight_vector + (alpha / sigma) * total_fitness_player2
        torch.nn.utils.vector_to_parameters(weight_vector, temp_player.model.parameters())
        self.population_player2[0] = temp_player.get_weights()
        del temp_player
        # #update weights of player2
        # fitness_player2 = np.array(fitness_player2)
        # epsilons_mutations2 = np.array(epsilons_mutations2)
        #
        # total_fitness_player2 = np.sum(np.multiply(fitness_player2, epsilons_mutations2.T), axis=0)/(len(fitness_player2)*len(fitness_player2[0]))
        # f1 = torch.FloatTensor([(alpha / sigma) * np.sum(total_fitness_player2)])
        # for key in self.population_player2[0].keys():
        #     self.population_player2[0][key] = self.population_player2[0][key] + f1

        team_fit = self.worker_class.evaluate_team_fitness(self.population_player1[0], self.population_player2[0])

        #update hall of fame player1
        best_weights_player1 = np.argmax(fitness_player1, axis=1)
        for i in range(len(best_weights_player1)):
            self.player1_hof.append(weights_mutations1[i][best_weights_player1[i]])

        #update hall of fame player2
        best_weights_player2 = np.argmax(fitness_player2, axis=1)
        for i in range(len(best_weights_player2)):
            self.player2_hof.append(weights_mutations2[i][best_weights_player2[i]])

        if team_fit > 50:
            torch.save(self.population_player1[0], "Models/best_weights_player1.pt")
            torch.save(self.population_player2[0], "Models/best_weights_player2.pt")
            print("saved")

        summary= {"team_fit": team_fit, "generation": self.generation}

        self.increment_metrics()
        return summary

    def compute_weight_update(self, noises, normalized_rewards):
        """ Compute the weight update using the update rule from the OpenAI ES. """
        config = self.config
        factor = config['learning_rate'] / (
                config['population_size'] * config['mutation_power'])
        weight_update = factor * np.dot(np.array(noises).T, normalized_rewards)
        return weight_update

    def normalize_rewards(self, rewards):
        """ Normalize the rewards using z-normalization. """
        rewards = np.array(rewards)
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        if reward_std == 0:
            return rewards - reward_mean
        else:
            return (rewards - reward_mean) / reward_std

    def step_hof(self):
        """ Evolve one generation using the Evolution Strategies algorithm.
        This consists of four steps:
        1. Send the current weights to a number of workers and mutate and evaluate them.
        2. Communicate the mutated weights and their fitness back to the Trainer.
        3. Update the weights using the ES update rule.
        4. Evaluate the updated weights against a random policy and log the outcome.
        """
        mutation_pool = Pool(processes=self.n)
        mutations_player1 = []
        mutations_player2 = []
        # self.worker_class.mutate_individual(self.player1_hof[0])
        for i in range(len(self.player1_hof_best)):
            temp = [copy.deepcopy(self.player1_hof_best[i])] * self.n
            mutations_player1 += [mutation_pool.map(self.worker_class.mutate_individual, temp)]
        for i in range(len(self.player2_hof_best)):
            temp = [copy.deepcopy(self.player2_hof_best[i])] * self.n
            mutations_player2 += [mutation_pool.map(self.worker_class.mutate_individual, temp)]
        mutation_pool.close()
        weights_mutations1 = []
        weights_mutations2 = []
        epsilons_mutations1 = []
        epsilons_mutations2 = []
        for i in range(len(mutations_player1)):
            wm, em = zip(*mutations_player1[i])
            weights_mutations1 += [wm]
            epsilons_mutations1 += [em]
        for i in range(len(mutations_player2)):
            wm, em = zip(*mutations_player2[i])
            weights_mutations2 += [wm]
            epsilons_mutations2 += [em]
        # self.worker_class.evaluate_team_fitness(weights_mutations1[0][2], self.player2_hof_best[0])
        fitness_pool = Pool(processes=self.n)
        #player1 fitness evaluation
        fitness_player1 = []
        for i in range(len(weights_mutations1)):
            fitness_player1 += [fitness_pool.starmap(self.worker_class.evaluate_player_fitness, zip(weights_mutations1[i], [self.player2_hof_best[0]]*self.n))]

        #player2 fitness evaluation
        fitness_player2 = []
        for i in range(len(weights_mutations2)):
            fitness_player2 += [fitness_pool.starmap(self.worker_class.evaluate_player_fitness, zip( [self.player1_hof_best[0]]*self.n, weights_mutations2[i]))]

        fitness_pool.close()
        alpha = 0.0001
        sigma = 0.1
        #update weights of player1
        fitness_player1 = np.array(fitness_player1)
        epsilons_mutations1 = np.array(epsilons_mutations1)


        # total_fitness_player1 = np.sum(np.multiply(fitness_player1, epsilons_mutations1), axis=1)/(len(fitness_player1)*len(fitness_player1[0]))
        # f1 = torch.FloatTensor([(alpha / sigma) * np.sum(total_fitness_player1)])
        temp_player = VBNChromosome(number_actions=self.config['number_actions'])
        temp_player.set_weights(self.population_player1[0])
        total_fitness_player1 = 0
        for i in range(len(fitness_player1)):
            total_fitness_player1 += np.multiply(fitness_player1[i],epsilons_mutations1[i].transpose()).transpose().sum()
        total_fitness_player1 = total_fitness_player1/(len(fitness_player1)*len(fitness_player1[0]))
        weight_vector = torch.nn.utils.parameters_to_vector(temp_player.model.parameters())
        weight_vector = weight_vector + (alpha / sigma) * total_fitness_player1
        torch.nn.utils.vector_to_parameters(weight_vector, temp_player.model.parameters())
        self.population_player1[0] = temp_player.get_weights()
        del temp_player
        #update weights of player2

        fitness_player2 = np.array(fitness_player2)
        epsilons_mutations2 = np.array(epsilons_mutations2)
        # total_fitness_player2 = np.sum(np.multiply(fitness_player2, epsilons_mutations2), axis=1)/(len(fitness_player2)*len(fitness_player2[0]))
        # f2 = torch.FloatTensor([(alpha / sigma) * np.sum(total_fitness_player2)])
        temp_player = VBNChromosome(number_actions=self.config['number_actions'])
        temp_player.set_weights(self.population_player2[0])
        total_fitness_player2 = 0
        for i in range(len(fitness_player2)):
            total_fitness_player2 += np.multiply(fitness_player2[i],epsilons_mutations2[i].transpose()).transpose().sum()
        total_fitness_player2 = total_fitness_player2/(len(fitness_player2)*len(fitness_player2[0]))
        weight_vector = torch.nn.utils.parameters_to_vector(temp_player.model.parameters())
        weight_vector = weight_vector + (alpha / sigma) * total_fitness_player2
        torch.nn.utils.vector_to_parameters(weight_vector, temp_player.model.parameters())
        self.population_player2[0] = temp_player.get_weights()
        del temp_player
        # #update weights of player2
        # fitness_player2 = np.array(fitness_player2)
        # epsilons_mutations2 = np.array(epsilons_mutations2)
        #
        # total_fitness_player2 = np.sum(np.multiply(fitness_player2, epsilons_mutations2.T), axis=0)/(len(fitness_player2)*len(fitness_player2[0]))
        # f1 = torch.FloatTensor([(alpha / sigma) * np.sum(total_fitness_player2)])
        # for key in self.population_player2[0].keys():
        #     self.population_player2[0][key] = self.population_player2[0][key] + f1

        player1_fit= self.worker_class.evaluate_player_fitness(self.population_player1[0], self.population_player2[0])
        player2_fit = self.worker_class.evaluate_player_fitness(self.population_player2[0], self.population_player1[0])

        #update hall of fame player1
        best_weights_player1 = np.argmax(fitness_player1, axis=1)
        for i in range(len(best_weights_player1)):
            self.player1_hof.append(weights_mutations1[i][best_weights_player1[i]])

        #update hall of fame player2
        best_weights_player2 = np.argmax(fitness_player2, axis=1)
        for i in range(len(best_weights_player2)):
            self.player2_hof.append(weights_mutations2[i][best_weights_player2[i]])

        if (player1_fit+player2_fit) > 50:
            torch.save(self.population_player1[0], "Models/best_weights_player1.pt")
            torch.save(self.population_player2[0], "Models/best_weights_player2.pt")
            print("saved")

        summary= {"player1_fit": player1_fit, "player2_fit":player2_fit, "generation": self.generation}

        self.increment_metrics()
        return summary
