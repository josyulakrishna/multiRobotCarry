import time

import gym
import os
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from MotionPlanningEnv.urdfObstacle import UrdfObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from urdfenvs.sensors.lidar import Lidar
import matplotlib.pyplot as plt

plt.ion()
def get_bc_features(ob):
    goal_position = np.array([8., 0.0, 0.])
    robot_positions = np.zeros((2, 3))
    for i, key in enumerate(ob.keys()):
        #get robot position
        robot_positions[i,:] = ob[key]['joint_state']['position']
    robot_centroid = robot_positions.mean(axis=0)
    slope_goal = (goal_position[1] - robot_centroid[1])/(goal_position[0] - robot_centroid[0])
    slope_robots = (robot_positions[1][1] - robot_positions[0][1])/((robot_positions[1][0] - robot_positions[0][0])+0.001)
    return np.linalg.norm(robot_centroid - goal_position), np.linalg.norm(robot_positions[0,:] - robot_positions[1,:] ), slope_goal, slope_robots



def dist_goal_robots(ob):
    goal_position = np.array([8., 0.0, 0.])
    robot_positions = np.zeros((2, 3))
    for i, key in enumerate(ob.keys()):
        # get robot position
        robot_positions[i, :] = ob[key]['joint_state']['position']
    robot_centroid = robot_positions.mean(axis=0)
    return np.linalg.norm(robot_centroid - goal_position), np.linalg.norm(robot_positions[0, :] - robot_positions[1, :])


lidar = Lidar(4, nb_rays=4, raw_data=False)
urdf_obstacle_dict = {
    "type": "urdf",
    "geometry": {"position": [0.2, -0.0, 1.05]},
    "urdf": os.path.join(os.path.dirname(__file__), "block.urdf"),
}
urdf_obstacle = UrdfObstacle(name="carry_object", content_dict=urdf_obstacle_dict)
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

def add_stuff(env):
    env.add_obstacle(urdf_obstacle)
    env.add_goal(goal)
    env.add_sensor(lidar, robot_ids=[0, 1])
    env.add_walls()



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


def make_env(render=False):
    robots = [
        GenericUrdfReacher(urdf="/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/loadPointRobot.urdf", mode="vel"),
        GenericUrdfReacher(urdf="/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/loadPointRobot.urdf", mode="vel"),
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

def kill_env(env):
    env.close()
    del env

def run_multi_robot_carry(n_steps=1000, render=False):
    # Two robots to carry the load

    history = []
    i = 0
    done = False
    robot_positions = np.zeros((2, 2))
    # env, ob = make_env(render=render)
    envs = gym.vector.make("urdf-env-v1", num_envs=3)
    # action = [0.1, 0.0, 0.0, 0.1, 0.0, 0.0]
    # action = [0.01967737, 0.14547452, 0., -0.01090256, -0.18248633, 0.]
    action = [-0.13163272,  0.02828632,  0.0, -0.03071381,  0.06311578, 0.0]
    goal_pos = np.array([8., 0.0])
    for _ in range(n_steps):
        i += 1
        # print("i ,", i)
        if done==True:
            kill_env(env)
            env, ob = make_env(render=render)
            r1, r2 = dist_goal_robots(ob)
            print(ob)
            done = False
        else:
            # WARNING: The reward function is not defined for you case.
            # You will have to do this yourself.
            ob, reward, done, info = env.step(action)
            print(done)
            dist2goal, distbrobots, slopegoal, sloperobots = get_bc_features(ob)
            for i, key in enumerate(ob.keys()):
                # get robot position
                robot_positions[i, :] = ob[key]['joint_state']['position'][:2]
            area_tri = abs(np.cross(robot_positions[0, :] - robot_positions[1, :], goal_pos - robot_positions[0, :]))
            area_tri = np.linalg.norm(area_tri) / 2
            # r1, r2 = dist_goal_robots(ob)

            plt.plot( i, area_tri, 'r+')
            plt.plot(i, dist2goal, 'b+')
            if info["goal_reached"]==True:

                print("Goal reached")
        # if done:
        #     break
        history.append(ob)
    env.close()

    return history


if __name__ == "__main__":
    run_multi_robot_carry(render=True)
