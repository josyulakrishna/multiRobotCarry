from envs_repo.gym_wrapper import GymWrapper
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import gym
import numpy as np

class EnvConstructor:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, env_name, frameskip):
        """
        A general Environment Constructor
        """
        self.env_name = env_name
        self.frameskip = frameskip

        #Dummy env to get some macros
        # dummy_env = self.make_env(render=False)
        self.is_discrete = False
        self.state_dim = 28 #dummy_env.state_dim
        self.action_dim = 4 #dummy_env.action_dim

    def make_env(self, render=False):
        robots = [
            GenericUrdfReacher(urdf="/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/loadPointRobot.urdf",
                               mode="vel"),
            GenericUrdfReacher(urdf="/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/loadPointRobot.urdf",
                               mode="vel"),
        ]
        env = gym.make("urdf-env-v1", dt=0.1, robots=robots, render=render)
        # Choosing arbitrary actions
        base_pos = np.array(
            [
                [0.0, 0.75, 0.0],
                [0.0, -0.75, 0.0],
            ]
        )
        env.reset(base_pos=base_pos)
        env.add_stuff()
        ob = env.get_observation()
        return env, ob



