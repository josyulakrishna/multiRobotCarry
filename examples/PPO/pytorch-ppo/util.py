import numpy as np
import gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher


pathLoad = "/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/loadPointRobot.urdf"
def get_gym_env_info(env_name):
    robots = [
        GenericUrdfReacher(urdf=pathLoad, mode="vel"),
        GenericUrdfReacher(urdf=pathLoad, mode="vel"),
    ]
    env = gym.make(env_name, dt=0.01, robots=robots, render=False, flatten_observation=False)

    base_pos = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )

    env.reset(base_pos=base_pos)
    env.add_stuff()

    # obs_shape = env.observation_space.shape
    obs_shape = env.observation_spaces_ppo()
    num_obs = int(np.product(obs_shape))

    try:
        # discrete space
        num_actions = env.action_space.n
        action_type = "discrete"
    except AttributeError:
        # continuous space
        num_actions = env.action_spaces_ppo().shape[0]
        action_type = "continuous"
    return num_actions, obs_shape, num_obs, action_type

if __name__ == "__main__":
    print("run")
    num_actions, obs_shape, num_obs, action_type = get_gym_env_info("urdf-env-v0")
    print("terminate")
    print(num_actions, obs_shape, num_obs, action_type)