from itertools import chain
from random import random
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
# torch.manual_seed(1836)
class VBNChromosome:
    """ Class that wraps the neural network. Includes functionality
    for Virtual Batch Normalization and the mutation of weights."""

    def __init__(self, number_actions=3):
        self.number_actions = number_actions
        self.observation_space = 14
        model = self.construct_layers()
        model.apply(self.weights_init_uniform_rule)
        self.model = model

    def construct_layers(self):
        model = nn.Sequential(
        nn.Linear(14, 64),
            nn.ReLU(),
         nn.Linear(64, 64),
            nn.ReLU(),
         nn.Linear(64, 3),
            # nn.Tanh()
        )

        return model

    # takes in a module and applies the specified weight initialization
    def weights_init_uniform_rule(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

    def virtual_batch_norm(self, samples):
        """ We apply Batch Normalization on a number of samples. By setting the learning
        rate to 0 we make sure that the weights and biases are not affected. This method
        is only ment to be used at the start of training."""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform([[x] for x in samples])
        return scaled

    def get_weights(self):
        """ Retrieve all the weights of the network. """
        return self.model.state_dict()

    def set_weights(self, new_weights, layers=None):
        """ Set all the weights of the network. """
        self.model.load_state_dict(new_weights)

    def mutate_weights(self):
        weights = self.mutate(1)
        self.model.load_state_dict(weights)

    def mutate(self, mutation_power):
        """ Mutate the current weights by adding a normally distributed vector of
        noise to the current weights. """
        weights = self.get_weights()
        for key in weights.keys():
            noise = np.random.normal(loc=0.0, scale=mutation_power, size=weights[key].shape)
            weights[key] = weights[key] + noise
        return weights

    def determine_actions(self, inputs):
        """ Choose an action based on the pixel inputs. We do this by simply
        selecting the action with the highest outputted value. """
        actions = self.model(inputs)
        return actions

if __name__ == "__main__":
    chromosome = VBNChromosome()
    chromosome.mutate(0.1)
    print("done")