import sys
import os
import numpy as np
import wandb

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from MotionPlanningEnv.urdfObstacle import UrdfObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from urdfenvs.sensors.lidar import Lidar
import pathlib
import yaml
import gym
from PPO2 import PPO
from datetime import datetime
import torch
from tracker import WandBTracker

# https://github.com/nikhilbarhate99/PPO-PyTorch


def make_env(render=False):
    robots = [
        GenericUrdfReacher(urdf="loadPointRobot.urdf", mode="vel"),
        GenericUrdfReacher(urdf="loadPointRobot.urdf", mode="vel"),
    ]
    env = gym.make("urdf-env-v0", dt=0.1, robots=robots, render=render)
    # Choosing arbitrary actions
    base_pos = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    env.reset(base_pos=base_pos)
    env.add_stuff()
    ob = env.get_observation()
    return env, ob
def kill_env(env):
    env.close()
    del env

def flatten_observation(observation_dictonary: dict) -> np.ndarray:

    observation_list = []
    for val in observation_dictonary.values():
        if isinstance(val, np.ndarray):
            observation_list += val.tolist()
        elif isinstance(val, dict):
            observation_list += flatten_observation(val).tolist()
    observation_array = np.array(observation_list)
    return observation_array



def train(render=False):
    # Two robots to carry the load
    env,_ = make_env(render=render)
    # history = []
    # for _ in range(n_steps):
    #     # WARNING: The reward function is not defined for you case.
    #     # You will have to do this yourself.
    #     ob, reward, done, info = env.step(action)
    #     print(ob)
    #     history.append(ob)
    # env.close()
    # return history
    env_name = "MultiRobotCarry"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len/4 # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 42        # set random seed if required (0 = no random seed)
    #####################################################
    wandb.init(project="ppo_dec_multirobot_carry",
    config=
        {
            "env_name": env_name,
            "has_continuous_action_space": has_continuous_action_space,
            "max_ep_len": max_ep_len,
            "max_training_timesteps": max_training_timesteps,
            "print_freq": print_freq,
            "log_freq": log_freq,
            "save_model_freq": save_model_freq,
            "action_std": action_std,
            "action_std_decay_rate": action_std_decay_rate,
            "min_action_std": min_action_std,
            "action_std_decay_freq": action_std_decay_freq,
            "update_timestep": update_timestep,
            "K_epochs": K_epochs,
            "eps_clip": eps_clip,
            "gamma": gamma,
            "lr_actor": lr_actor,
            "lr_critic": lr_critic,
            "random_seed": random_seed,
        }
    )
    print("training environment name : " + env_name)
    # state space dimension
    state_dim = env.observation_spaces_ppo().shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_spaces_ppo().shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################
    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        # env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    # initialize a PPO agent
    ppo_agent_1 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
    ppo_agent_2 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:
        kill_env(env)
        env, state = make_env(render=False)
        state_0 = np.append(flatten_observation(state['robot_0']), state['robot_1']['joint_state']["position"])
        state_1 = np.append(flatten_observation(state['robot_1']), state['robot_0']['joint_state']["position"])
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action1 = ppo_agent_1.select_action(state_0)
            action2 = ppo_agent_2.select_action(state_1)
            actions = np.concatenate((action1, np.zeros(1), action2, np.zeros(1)))
            state, reward, done, _ = env.step(actions)

            state_0 = np.append(flatten_observation(state['robot_0']), state['robot_1']['joint_state']["position"])
            state_1 = np.append(flatten_observation(state['robot_1']), state['robot_0']['joint_state']["position"])

            # saving reward and is_terminals
            ppo_agent_1.buffer.rewards.append(reward[0])
            ppo_agent_1.buffer.is_terminals.append(done)

            ppo_agent_2.buffer.rewards.append(reward[0])
            ppo_agent_2.buffer.is_terminals.append(done)


            time_step += 1
            current_ep_reward = reward[0]+reward[1]

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent_1.update()
                ppo_agent_2.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent_1.decay_action_std(action_std_decay_rate, min_action_std)
                ppo_agent_2.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()
                wandb.log({'log_avg_reward': log_avg_reward, 'time_step': time_step, 'i_episode': i_episode})
                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,print_avg_reward))
                wandb.log({'episode': i_episode, 'time_step': time_step, 'avg_reward': print_avg_reward})
                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent_1.save(checkpoint_path+'_1')
                ppo_agent_2.save(checkpoint_path+'_2')
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    kill_env(env)
    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()

