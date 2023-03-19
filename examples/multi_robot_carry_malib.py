import gym
import os
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from MotionPlanningEnv.urdfObstacle import UrdfObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from urdfenvs.sensors.lidar import Lidar
import pathlib
import yaml

BASE_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()

config_file = "marl.yaml"

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

training_config = config["trainer_config"]
rollout_config = config["rollout"]

assert (
        rollout_config["num_episodes"] % rollout_config["num_env_per_worker"] == 0
), "in rollout config (num_episodes mod epsidoe_seg) must be 0"
training_config["trainer_config"]["total_epoch"] = rollout_config["stopper_config"][
    "max_step"
]

custom_config = config["algorithms"]["MAPPO"]["custom_config"]
custom_config["use_popart"] = False
custom_config["use_feature_normalization"] = False
# FOR MAPPO
custom_config.update({"global_state_space": None})
model_config = config["algorithms"]["MAPPO"]["model_config"]


# evaluation_config = config["evaluation"]
# env_desc = env_desc_gen(config["env_description"]["config"])

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

def create_obstacle():
    """
    You can place any arbitrary urdf file in here.
    The physics of this box is not perfect, but this is not the field of
    my expertise.
    """

    urdf_obstacle_dict = {
        "type": "urdf",
        "geometry": {"position": [0.2, -0.0, 1.05]},
        "urdf": os.path.join(os.path.dirname(__file__), "block.urdf"),
    }
    urdf_obstacle = UrdfObstacle(name="carry_object", content_dict=urdf_obstacle_dict)
    return urdf_obstacle


def run_multi_robot_carry(n_steps=1000, render=False):
    # Two robots to carry the load
    robots = [
        GenericUrdfReacher(urdf="loadPointRobot.urdf", mode="vel"),
        GenericUrdfReacher(urdf="loadPointRobot.urdf", mode="vel"),
    ]
    env = gym.make("urdf-env-v0", dt=0.01, robots=robots, render=render)
    # Choosing arbitrary actions
    action = [0.2, 0.0, 0.0, 0.2, -0.5, 0.0]
    base_pos = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    ob = env.reset(base_pos=base_pos)
    # Placing the obstacle
    env.add_obstacle(create_obstacle())
    env.add_goal(goal)
    lidar = Lidar(4, nb_rays=4, raw_data=False)
    env.add_sensor(lidar, robot_ids=[0, 1])
    env.add_walls()

    history = []

    possible_agents = env.possible_agents()
    action_spaces = env.possible_actions()
    observation_spaces = env.observation_spaces()
    env_desc = {"creator": env, "possible_agents": possible_agents, "action_spaces": action_spaces, "observation_spaces": observation_spaces}


    for _ in range(n_steps):
        # WARNING: The reward function is not defined for you case.
        # You will have to do this yourself.
        ob, reward, done, info = env.step(action)
        # print(done)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_multi_robot_carry(render=True)
