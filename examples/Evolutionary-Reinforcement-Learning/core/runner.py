
from core import utils as utils
import numpy as np
import torch


def flatten_observation(observation_dictonary: dict) -> np.ndarray:
    observation_list = []
    for val in observation_dictonary.values():
        if isinstance(val, np.ndarray):
            observation_list += val.tolist()
        elif isinstance(val, dict):
            observation_list += flatten_observation(val).tolist()
    observation_array = np.array(observation_list)
    return observation_array

# Rollout evaluate an agent in a complete game
@torch.no_grad()
def rollout_worker(id, type, task_pipe, result_pipe, store_data, model_bucket, env_constructor):



    env = env_constructor.make_env()
    np.random.seed(id) ###make sure the random seeds across learners are different

    ###LOOP###
    while True:
        identifier = task_pipe.recv()  # Wait until a signal is received  to start rollout
        if identifier == 'TERMINATE': exit(0) #Exit

        # Get the requisite network
        net = model_bucket[identifier]

        fitness = 0.0
        total_frame = 0
        env, state = env_constructor.make_env(render=False)
        rollout_trajectory = []
        state = flatten_observation(state)
        state = utils.to_tensor(state)
        while True:  # unless done

            if type == 'pg': action = net.noisy_action(state)
            else: action = net.clean_action(state)

            action = utils.to_numpy(action)
            next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment
            next_state = flatten_observation(next_state)
            next_state = utils.to_tensor(next_state)
            fitness += reward[0]+reward[1]

            # If storing transitions
            if store_data: #Skip for test set
                rollout_trajectory.append([utils.to_numpy(state), utils.to_numpy(next_state),
                                        np.float32(action), np.reshape(np.float32(np.array([reward])), (1, 1)),
                                           np.reshape(np.float32(np.array([float(done)])), (1, 1))])
            state = next_state
            total_frame += 1

            # DONE FLAG IS Received
            if done:
                break

        # Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([identifier, fitness, total_frame, rollout_trajectory])
