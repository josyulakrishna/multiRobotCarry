from mappo.mappo_trainer import MAPPOTrainer
from mappo.ppo_model import PolicyNormal
from mappo.ppo_model import CriticNet
from mappo.ppo_model import ActorNet
from mappo.ppo_agent import PPOAgent
import numpy as np
import torch
import gym
import os
from urdfenvs.robots.generic_urdf import GenericUrdfReacher

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

def load_env(render=False):
    """
    Initializes the UnityEnviornment and corresponding environment variables
    based on the running operating system.

    Arguments:
        env_loc: A string designating unity environment directory.

    Returns:
        env: A UnityEnvironment used for Agent evaluation and training.
        num_agents: Integer number of Agents to be trained in environment.
        state_size: Integer number of possible states.
        action_size: Integer number of possible actions.
    """

    env,ob = make_env(render=render)

    # Extract state dimensionality from env.
    state_size = env.observation_spaces_maddpg()[0].shape[0]

    # Extract action dimensionality and number of agents from env.
    action_size = env.action_spaces_maddpg()[0].shape[0]
    num_agents = env.n_

    # Display relevant environment information.
    print('\nNumber of Agents: {}, State Size: {}, Action Size: {}\n'.format(
        num_agents, state_size, action_size))

    return env, num_agents, state_size, action_size


def create_agent(state_size, action_size, actor_fc1_units=512,
                 actor_fc2_units=256, actor_lr=1e-4, critic_fc1_units=512,
                 critic_fc2_units=256, critic_lr=1e-4, gamma=0.99,
                 num_updates=10, max_eps_length=500, eps_clip=0.3,
                 critic_loss=0.5, entropy_bonus=0.01, batch_size=256):
    """
    This function creates an agent with specified parameters for training.

    Arguments:
        state_size: Integer number of possible states.
        action_size: Integer number of possible actions.
        actor_fc1_units: An integer number of units used in the first FC
            layer for the Actor object.
        actor_fc2_units: An integer number of units used in the second FC
            layer for the Actor object.
        actor_lr: A float designating the learning rate of the Actor's
            optimizer.
        critic_fc1_units: An integer number of units used in the first FC
            layer for the Critic object.
        critic_fc2_units: An integer number of units used in the second FC
            layer for the Critic object.
        critic_lr: A float designating the learning rate of the Critic's
            optimizer.
        gamma: A float designating the discount factor.
        num_updates: Integer number of updates desired for every
            update_frequency steps.
        max_eps_length: An integer for maximum number of timesteps per
            episode.
        eps_clip: Float designating range for clipping surrogate objective.
        critic_loss: Float designating initial Critic loss.
        entropy_bonus: Float increasing Actor's tendency for exploration.
        batch_size: An integer for minibatch size.

    Returns:
        agent: An Agent object used for training.
    """

    # Create Actor/Critic networks based on designated parameters.
    actor_net = ActorNet(state_size, action_size, actor_fc1_units,
                         actor_fc2_units).to(device)
    critic_net = CriticNet(state_size, critic_fc1_units, critic_fc2_units)\
        .to(device)

    # Create copy of Actor/Critic networks for action prediction.
    actor_net_old = ActorNet(state_size, action_size, actor_fc1_units,
                             actor_fc2_units).to(device)
    critic_net_old = CriticNet(state_size, critic_fc1_units, critic_fc2_units)\
        .to(device)
    actor_net_old.load_state_dict(actor_net.state_dict())
    critic_net_old.load_state_dict(critic_net.state_dict())

    # Create PolicyNormal objects containing both sets of Actor/Critic nets.
    actor_critic = PolicyNormal(actor_net, critic_net)
    actor_critic_old = PolicyNormal(actor_net_old, critic_net_old)

    # Initialize optimizers for Actor and Critic networks.
    actor_optimizer = torch.optim.Adam(
        actor_net.parameters(),
        lr=actor_lr
    )
    critic_optimizer = torch.optim.Adam(
        critic_net.parameters(),
        lr=critic_lr
    )

    # Create and return PPOAgent with relevant parameters.
    agent = PPOAgent(
        device=device,
        actor_critic=actor_critic,
        actor_critic_old=actor_critic_old,
        gamma=gamma,
        num_updates=num_updates,
        eps_clip=eps_clip,
        critic_loss=critic_loss,
        entropy_bonus=entropy_bonus,
        batch_size=batch_size,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer
    )

    return agent


def create_trainer(env, agents, save_dir, update_frequency=5000,
                   max_eps_length=500, score_window_size=100):
    """
    Initializes trainer to train agents in specified environment.

    Arguments:
        env: A UnityEnvironment used for Agent evaluation and training.
        agents: Agent objects used for training.
        save_dir: Path designating directory to save resulting files.
        update_frequency: An integer designating the step frequency of
            updating target network parameters.
        max_eps_length: An integer for maximum number of timesteps per
            episode.
        score_window_size: Integer window size used in order to gather
            max mean score to evaluate environment solution.

    Returns:
        trainer: A MAPPOTrainer object used to train agents in environment.
    """

    # Initialize MAPPOTrainer object with relevant arguments.
    trainer = MAPPOTrainer(
        env=env,
        agents=agents,
        score_window_size=score_window_size,
        max_episode_length=max_eps_length,
        update_frequency=update_frequency,
        save_dir=save_dir
    )

    return trainer


def test_agents(env, trainer):
    """
    This function carries out the training process with specified trainer.

    Arguments:
        env: A UnityEnvironment used for Agent evaluation and training.
        trainer: A MAPPOTrainer object used to train agents in environment.
        n_episodes: An integer for maximum number of training episodes.
        target_score: An float max mean target score to be achieved over
            the last score_window_size episodes.
        score_window_size: The integer number of past episode scores
            utilized in order to calculate the current mean scores.
    """

    actions = []
    goal = 0
    for i in range(1000):
        kill_env(env)
        env, ob = make_env(render=False)
        obs = flatten_observation(ob)
        obs_list = obs.reshape(env.n_, 14)
        states = obs_list
        done = False
        while not done:
            processed_states, actions, log_probs = [], [], []
            for agent, state in zip(trainer.agents, states):
                processed_state = torch.from_numpy(state).float()
                action, log_prob = agent.get_actions(processed_state)
                actions.append(action)
            raw_actions = np.array(
                [torch.clamp(a, -1, 1).numpy() for a in actions]
            )
            raw_actions = np.hstack((raw_actions, np.zeros((int(env.n_), 1)))).ravel()
            states, rewards, done, info = env.step(raw_actions)
            states = flatten_observation(states)
            states = states.reshape(env.n_, 14)
            dones = [done]* env.n_
            done = any(dones)
            if info['goal_reached']:
                print('goal reached')
                goal += 1
            else:
                print('goal not reached')
            # print("rewards = ", rewards)
    kill_env(env)



if __name__ == '__main__':

    # Initialize environment, extract state/action dimensions and num agents.
    env, num_agents, state_size, action_size = load_env(render=False)

    # Initialize agents for training.
    agents = [create_agent(state_size, action_size) for _ in range(num_agents)]

    # Create MAPPOTrainer object to train agents.
    save_dir = os.path.join(os.getcwd(), r'saved_files')
    trainer = create_trainer(env, agents, save_dir, update_frequency=1000, max_eps_length=500)
    # Train agent in specified environment.
    # train_agents(env, trainer)
    paths= ["/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/mappov1/saved_files/run1/agent_0_episode_3881.pth",
            "/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/mappov1/saved_files/run1/agent_1_episode_3881.pth"]

    trainer.load_state(paths)
    test_agents(env, trainer)
