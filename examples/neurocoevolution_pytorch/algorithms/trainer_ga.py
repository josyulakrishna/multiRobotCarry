# from pathlib import Path
import random
from collections import deque, defaultdict
from functools import partial
import numpy as np
# import ray
import yaml
# from ray.rllib.agents import with_common_config
from torch.multiprocessing import Pool
from algorithms.trainer_ea import EATrainer
from utils_torch.chromosome import VBNChromosome
import copy
import pickle

DEFAULT_CONFIG = {}
import wandb
# wandb.init(project="neurocoevolution_mr_carry_hof_diff", entity="josyula", sync_tensorboard=True, config=DEFAULT_CONFIG)
with open('configs/config_ga_test.yaml') as f:
    DEFAULT_CONFIG = yaml.load(f, Loader=yaml.FullLoader)



class GATrainer(EATrainer):
    _name = "GA"
    _default_config = DEFAULT_CONFIG

    def __init__(self, config):
        """ Trainer class for the Coevolutionary Genetic Algorithm.
        This class distributes the mutation and evaluation workload over a number
        of workers and updates and maintains the population."""

        super(GATrainer, self).__init__(config)
        self.M = self.config['population_pool']
        self.k = self.config['population_size']
        chromosome = VBNChromosome(number_actions=self.config['number_actions'])
        self.population_player1 = [
            [ chromosome.get_weights() for _ in range(self.k)]
            for _ in range(self.M)

        ]

        self.population_player2 = [
            [chromosome.get_weights() for _ in range(self.k)]
            for _ in range(self.M)

        ]



        self.player1_hof = deque(maxlen=self.M)
        self.player2_hof = deque(maxlen=self.M)
        self.prev_fitness = deque(maxlen=10)
        self.player1_hof_prev = deque(maxlen=10)
        self.player2_hof_prev = deque(maxlen=10)
        self.rest_mutation_rate = False
        # self.elites = [VBNChromosome(number_actions=self.config['number_actions'])
        #                for _ in range(config['number_elites'])]
        # samples = self.collect_samples()
        # for chrom in self.elites:
        #     chrom.virtual_batch_norm(samples)

        # self.hof = [self.elites[i].get_weights() for i in
        #             range(self.config['number_elites'])]
        # self.winner = None
        self.summary = {"prev_gen_fit_1": -1000,
                        "prev_gen_fit_2": -1000}


    def step_hof(self):
        """ Evolve the next generation using the Genetic Algorithm. This process
        1 Begining of the algorithm construct M populations(population_pool) of size k (population_size)
        2 Mutate the k individuals in each population 2k individuals (can be parallelized)
        3 Form teams of 2 individuals from each population and play a game (can be parallelized)
        4 Evaluate fitness of each team, assign team fitness to each individual
        5 Select k individuals from each population to form the next generation
        6 Repeat from step 2 until the desired number of generations is reached
        """

        # step2 mutate the individuals in each population
        previous_generation_player1 = copy.deepcopy(self.population_player1)
        previous_generation_player2 = copy.deepcopy(self.population_player2)
        mutation_pool = Pool(processes=self.k)
        mutations_player1 = []
        mutations_player2 = []
        if self.summary["prev_gen_fit_1"] > 0 :
            self.worker_class.mutation_power = 0.01
        for i in range(self.M):
            mutations_player1 += [mutation_pool.map(self.worker_class.mutate_individual, self.population_player1[i])]
        for i in range(self.M):
            mutations_player2 += [mutation_pool.map(self.worker_class.mutate_individual, self.population_player2[i])]
        mutation_pool.close()

        if self.rest_mutation_rate:
            self.worker_class.mutation_power = self.config['mutation_power']
            #replace a few individuals with the previous generation randomly
            mutations_player1[np.random.randint(0, self.M)][0] = self.player1_hof_prev[np.random.randint(0, len(self.player1_hof_prev))]
            mutations_player1[np.random.randint(0, self.M)][1] = self.player1_hof_prev[np.random.randint(0, len(self.player1_hof_prev))]
            mutations_player2[np.random.randint(0, self.M)][0] = self.player2_hof_prev[np.random.randint(0, len(self.player2_hof_prev))]
            mutations_player2[np.random.randint(0, self.M)][1] = self.player2_hof_prev[np.random.randint(0, len(self.player2_hof_prev))]
            self.rest_mutation_rate = False
        # step 3 form teams of 2 individuals from each population and play a game
        # evaluate individuals against best in other populations
        # start with player1
        # randomly individual from population of player2 if no best individuals of player2 yet
        fitness_pool = Pool(processes=2*self.k)
        fitness_pop_player1_prev_gen = []
        fitness_pop_player1_mut_gen = []
        if len(self.player2_hof) == 0:
            inds = np.random.randint(0, self.k, self.M).tolist() #selects one individual from each population
            player2_individuals = [ mutations_player2[i][j] for i,j in zip(range(self.M),inds)]

            for i in range(self.M):
                fitness_pop_player1_prev_gen += [fitness_pool.starmap(self.worker_class.evaluate_team_fitness, zip(previous_generation_player1[i], [player2_individuals[i]]*self.k))]
            for i in range(self.M):
                fitness_pop_player1_mut_gen += [fitness_pool.starmap(self.worker_class.evaluate_team_fitness,
                                                             zip(mutations_player1[i],
                                                                 [player2_individuals[i]] * self.k))]
        else:
            for i in range(self.M):
                fitness_pop_player1_prev_gen += [fitness_pool.starmap(self.worker_class.evaluate_team_fitness, zip(previous_generation_player1[i], [self.player2_hof[i]]*self.k))]
            for i in range(self.M):
                fitness_pop_player1_mut_gen += [fitness_pool.starmap(self.worker_class.evaluate_team_fitness,
                                                             zip(mutations_player1[i],
                                                                 [self.player2_hof[i]] * self.k))]

        #player2
        fitness_pop_player2_prev_gen = []
        fitness_pop_player2_mut_gen = []
        if len(self.player1_hof) == 0:
            inds = np.random.randint(0, self.k, self.M).tolist()
            player1_individuals = [mutations_player1[i][j] for i, j in zip(range(self.M), inds)]
            for i in range(self.M):
                fitness_pop_player2_prev_gen += [fitness_pool.starmap(self.worker_class.evaluate_team_fitness, zip([player1_individuals[i]]*self.k, previous_generation_player2[i]))]
            for i in range(self.M):
                fitness_pop_player2_mut_gen += [fitness_pool.starmap(self.worker_class.evaluate_team_fitness,
                                                             zip([player1_individuals[i]] * self.k,
                                                                         mutations_player2[i]))]
        else:
            for i in range(self.M):
                fitness_pop_player2_prev_gen += [fitness_pool.starmap(self.worker_class.evaluate_team_fitness, zip([self.player1_hof[i]]*self.k, previous_generation_player2[i]))]
            for i in range(self.M):
                fitness_pop_player2_mut_gen += [fitness_pool.starmap(self.worker_class.evaluate_team_fitness,
                                                             zip([self.player1_hof[i]] * self.k,
                                                                         mutations_player2[i]))]
        fitness_pool.close()
        # replace the population with the new generation if the fitness is better
        ##player1##
        # hof_temp = defaultdict( partial(deque, maxlen=1))
        hof_temp = defaultdict(list)
        for i in range(self.M):
            for j in range(self.k):
                if fitness_pop_player1_prev_gen[i][j] < fitness_pop_player1_mut_gen[i][j]:
                    self.population_player1[i][j] = mutations_player1[i][j]
                    hof_temp[i].append(mutations_player1[i][j])
                else:
                    hof_temp[i].append(previous_generation_player1[i][j])


        #append the best individual of each population to the hall of fame
        for i in range(self.M):
            self.player1_hof.append(hof_temp[i][random.randint(0, len(hof_temp[i])-1)])

        ##player2##
        hof_temp = defaultdict(list)
        for i in range(self.M):
            for j in range(self.k):
                if fitness_pop_player2_prev_gen[i][j] < fitness_pop_player2_mut_gen[i][j]:
                    self.population_player2[i][j] = mutations_player2[i][j]
                    hof_temp[i].append(mutations_player2[i][j])
                else:
                    hof_temp[i].append(previous_generation_player2[i][j])
        #append the best individual of each population to the hall of fame
        for i in range(self.M):
            self.player2_hof.append(hof_temp[i][random.randint(0, len(hof_temp[i])-1)])


        # Evaluate all hofs
        hof_pool = Pool(processes=self.M)
        hof_fitness = hof_pool.starmap(self.worker_class.evaluate_team_fitness, zip(self.player1_hof, self.player2_hof))
        hof_pool.close()
        summary = {
            "generation":self.generation,
            "prev_gen_fit_1":np.mean(fitness_pop_player1_prev_gen),
            "mut_gen_fit_1":np.mean(fitness_pop_player1_mut_gen),
            "prev_gen_fit_2":np.mean(fitness_pop_player2_prev_gen),
            "mut_gen_fit_2":np.mean(fitness_pop_player2_mut_gen),
            "team_player1and2hofg":np.mean(hof_fitness),
            "goal_reaches":self.worker_class.goal_reach,
            "wall_pass":self.worker_class.wall_pass,
        }
        wandb.log(summary)
        if np.linalg.norm(np.mean(list(self.prev_fitness))- np.mean(hof_fitness)) < 0.01 and np.mean(hof_fitness)<0:
            self.worker_class.mutation_power = 0.5*self.worker_class.mutation_power
            self.reset_mutation_rate = True

        self.prev_fitness.append(hof_fitness)
        self.increment_metrics()
        if self.generation % 5 == 0:
            self.player1_hof_prev.append(self.player1_hof)
            self.player2_hof_prev.append(self.player2_hof)
        # self.add_videos_to_summary(results, summary)
        if np.mean(hof_fitness) > 0:
            with open("/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/neurocoevolution_pytorch/results/player1_hof.pkl", "w") as f:
                f.write(str(self.player1_hof))
            with open("/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/neurocoevolution_pytorch/results/player2_hof.pkl", "w") as f:
                f.write(str(self.player2_hof))
        return summary
###############################################################HOF DIFFERENCE###################################################
    def step_hof_difference_reward(self):

        # step2 mutate the individuals in each population
        previous_generation_player1 = copy.deepcopy(self.population_player1)
        previous_generation_player2 = copy.deepcopy(self.population_player2)
        mutation_pool = Pool(processes=self.k)
        mutations_player1 = []
        mutations_player2 = []
        if self.summary["prev_gen_fit_1"] > 0 :
            self.worker_class.mutation_power = 0.01

        for i in range(self.M):
            mutations_player1 += [mutation_pool.map(self.worker_class.mutate_individual, self.population_player1[i])]
        for i in range(self.M):
            mutations_player2 += [mutation_pool.map(self.worker_class.mutate_individual, self.population_player2[i])]
        mutation_pool.close()
        #
        # if self.rest_mutation_rate:
        #     self.worker_class.mutation_power = self.config['mutation_power']
        #     #replace a few individuals with the previous generation randomly
        #     mutations_player1[np.random.randint(0, self.M)][0] = self.player1_hof_prev[np.random.randint(0, len(self.player1_hof_prev))]
        #     mutations_player1[np.random.randint(0, self.M)][1] = self.player1_hof_prev[np.random.randint(0, len(self.player1_hof_prev))]
        #     mutations_player2[np.random.randint(0, self.M)][0] = self.player2_hof_prev[np.random.randint(0, len(self.player2_hof_prev))]
        #     mutations_player2[np.random.randint(0, self.M)][1] = self.player2_hof_prev[np.random.randint(0, len(self.player2_hof_prev))]
        #     self.rest_mutation_rate = False
        # step 3 Evaluate HOF difference rewards for each individual of each population
        # evaluate individuals against best in other populations
        # start with player1
        # randomly individual from population of player2 if no best individuals of player2 yet
        fitness_pool = Pool(processes=2*self.k)
        fitness_pop_player1_prev_gen = []
        fitness_pop_player1_mut_gen = []

        if len(self.player2_hof) == 0:
            inds = np.random.randint(0, self.k, self.M).tolist() #selects one individual from each population
            player2_individuals = [ mutations_player2[i][j] for i,j in zip(range(self.M),inds)]

            for i in range(self.M):
                fitness_pop_player1_prev_gen += [fitness_pool.starmap(self.worker_class.evaluate_hof_diff, zip(previous_generation_player1[i], [player2_individuals[i]]*self.k))]
            for i in range(self.M):
                fitness_pop_player1_mut_gen += [fitness_pool.starmap(self.worker_class.evaluate_hof_diff,
                                                             zip(mutations_player1[i],
                                                                 [player2_individuals[i]] * self.k))]
        else:
            for i in range(self.M):
                fitness_pop_player1_prev_gen += [fitness_pool.starmap(self.worker_class.evaluate_hof_diff, zip(previous_generation_player1[i], [self.player2_hof[i]]*self.k))]
            for i in range(self.M):
                fitness_pop_player1_mut_gen += [fitness_pool.starmap(self.worker_class.evaluate_hof_diff,
                                                             zip(mutations_player1[i],
                                                                 [self.player2_hof[i]] * self.k))]


        #player2
        fitness_pop_player2_prev_gen = []
        fitness_pop_player2_mut_gen = []
        if len(self.player1_hof) == 0:
            inds = np.random.randint(0, self.k, self.M).tolist()
            player1_individuals = [mutations_player1[i][j] for i, j in zip(range(self.M), inds)]
            for i in range(self.M):
                fitness_pop_player2_prev_gen += [fitness_pool.starmap(self.worker_class.evaluate_hof_diff, zip([player1_individuals[i]]*self.k, previous_generation_player2[i]))]
            for i in range(self.M):
                fitness_pop_player2_mut_gen += [fitness_pool.starmap(self.worker_class.evaluate_hof_diff,
                                                             zip([player1_individuals[i]] * self.k,
                                                                         mutations_player2[i]))]
        else:
            for i in range(self.M):
                fitness_pop_player2_prev_gen += [fitness_pool.starmap(self.worker_class.evaluate_hof_diff, zip([self.player1_hof[i]]*self.k, previous_generation_player2[i]))]
            for i in range(self.M):
                fitness_pop_player2_mut_gen += [fitness_pool.starmap(self.worker_class.evaluate_hof_diff,
                                                             zip([self.player1_hof[i]] * self.k,
                                                                         mutations_player2[i]))]
        fitness_pool.close()
        # replace the population with the new generation if the fitness is better
        ##player1##
        # hof_ind = np.zeros(self.M, )
        # for i in range(self.M):
        #     for j in range(self.k):
        #         if fitness_pop_player1_prev_gen[i][j] < fitness_pop_player1_mut_gen[i][j]:
        #             self.population_player1[i][j] = mutations_player1[i][j]
        #             hof_ind[i] = j
        # #append the best individual of each population to the hall of fame
        # for i in range(self.M):
        #     self.player1_hof.append(self.population_player1[i][int(hof_ind[i])])
        # ##player2##
        # for i in range(self.M):
        #     for j in range(self.k):
        #         if fitness_pop_player2_prev_gen[i][j] < fitness_pop_player2_mut_gen[i][j]:
        #             self.population_player2[i][j] = mutations_player2[i][j]
        #             hof_ind[i] = j
        # #append the best individual of each population to the hall of fame
        # for i in range(self.M):
        #     self.player2_hof.append(self.population_player2[i][int(hof_ind[i])])
        #

        hof_temp = defaultdict(list)
        for i in range(self.M):
            for j in range(self.k):
                if fitness_pop_player1_prev_gen[i][j] < fitness_pop_player1_mut_gen[i][j]:
                    self.population_player1[i][j] = mutations_player1[i][j]
                    hof_temp[i].append(mutations_player1[i][j])
                else:
                    hof_temp[i].append(previous_generation_player1[i][j])

        # append the best individual of each population to the hall of fame
        for i in range(self.M):
            self.player1_hof.append(hof_temp[i][random.randint(0, len(hof_temp[i]) - 1)])

        ##player2##
        hof_temp = defaultdict(list)
        for i in range(self.M):
            for j in range(self.k):
                if fitness_pop_player2_prev_gen[i][j] < fitness_pop_player2_mut_gen[i][j]:
                    self.population_player2[i][j] = mutations_player2[i][j]
                    hof_temp[i].append(mutations_player2[i][j])
                else:
                    hof_temp[i].append(previous_generation_player2[i][j])
        # append the best individual of each population to the hall of fame
        for i in range(self.M):
            self.player2_hof.append(hof_temp[i][random.randint(0, len(hof_temp[i]) - 1)])

        # Evaluate all hofs
        hof_pool = Pool(processes=self.M)
        hof_fitness = hof_pool.starmap(self.worker_class.evaluate_team_fitness, zip(self.player1_hof, self.player2_hof))

        hof_pool.close()
        summary = {
            "generation":self.generation,
            "prev_gen_fit_1":np.mean(fitness_pop_player1_prev_gen),
            "mut_gen_fit_1":np.mean(fitness_pop_player1_mut_gen),
            "prev_gen_fit_2":np.mean(fitness_pop_player2_prev_gen),
            "mut_gen_fit_2":np.mean(fitness_pop_player2_mut_gen),
            "team_player1and2hofg":np.mean(hof_fitness),
            "goal_reaches":self.worker_class.goal_reach,
            "wall_pass":self.worker_class.wall_pass,
        }
        wandb.log(summary)
        # if np.linalg.norm(np.mean(list(self.prev_fitness))- np.mean(hof_fitness)) < 0.01 and np.mean(hof_fitness)<0:
        #     self.worker_class.mutation_power = 0.5*self.worker_class.mutation_power
        #     self.reset_mutation_rate = True

        self.prev_fitness.append(hof_fitness)
        self.increment_metrics()
        if self.generation % 5 == 0:
            self.player1_hof_prev.append(self.player1_hof)
            self.player2_hof_prev.append(self.player2_hof)
        # self.add_videos_to_summary(results, summary)
        if np.mean(hof_fitness) > 0:
            with open("/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/neurocoevolution_pytorch/results/player1_hof.pkl", "w") as f:
                f.write(str(self.player1_hof))
            with open("/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/neurocoevolution_pytorch/results/player2_hof.pkl", "w") as f:
                f.write(str(self.player2_hof))
        return summary

    def timesteps(self):
        return self.timesteps_total