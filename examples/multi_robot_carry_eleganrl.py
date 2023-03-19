import sys
import os
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from MotionPlanningEnv.urdfObstacle import UrdfObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from urdfenvs.sensors.lidar import Lidar
import pathlib
import yaml
import gym
from stable_baselines3 import PPO

BASE_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()

goalDict = { "type": "static",
            "desired_position": [0.2, -0.0, 1.05],
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": 0,
            "child_link": 3,
            "epsilon": 0.2,
            "type": "staticSubGoal",
             }
goal = StaticSubGoal(name="goal", content_dict=goalDict)



def run_multi_robot_carry(n_steps=1000, render=False):
    # Two robots to carry the load
    robots = [
        GenericUrdfReacher(urdf="loadPointRobot.urdf", mode="vel"),
        GenericUrdfReacher(urdf="loadPointRobot.urdf", mode="vel"),
    ]
    env = gym.make("urdf-env-v0", dt=0.01, robots=robots, render=render)
    # # Choosing arbitrary actions
    action = [0.2, 0.0, 0.0, 0.2, -0.5, 0.0]
    base_pos = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )

    env.reset(base_pos=base_pos)
    # Placing the obstacle
    # env.add_obstacle(create_obstacle())
    # # env.add_goal(goal)
    # lidar = Lidar(4, nb_rays=4, raw_data=False)
    # env.add_sensor(lidar, robot_ids=[0, 1])
    # env.add_walls()
    env.add_stuff()
    # env_args = get_gym_env_args(env, if_print=True)
    # args = Arguments(agent_class, env_args=env_args, env_func=env_func)
    # train_and_evaluate(args)
    # print(args)

    history = []
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    for _ in range(n_steps):
        # WARNING: The reward function is not defined for you case.
        # You will have to do this yourself.
        ob, reward, done, info = env.step(action)
        print(ob)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_multi_robot_carry(render=True)
