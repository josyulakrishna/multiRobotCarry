import gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import numpy as np
from urdfenvs.sensors.lidar import Lidar
def run_point_robot(n_steps=1000, render=True, goal=False, obstacles=False):
    robots = [
        GenericUrdfReacher(urdf="multiPointRobot.urdf", mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.array([0.1, 0.0, 1.0, 0.1, 0.0, 1.0])
    pos0 = np.array([1.0, 0.1, 0.0, 1.5, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    lidar = Lidar(4, nb_rays=4, raw_data=False)
    ob = env.reset(pos=pos0, vel=vel0)
    env.add_sensor(lidar, robot_ids=[0, 1])

    # print(f"Initial observation : {ob['robot_0']['lidarSensor']}")
    print(ob['robot_0']['lidarSensor'])
    env.add_walls()

    if obstacles:
        from examples.scene_objects.obstacles import (
            sphereObst1,
            sphereObst2,
            urdfObst1,
            dynamicSphereObst3,
        )

        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
        env.add_obstacle(urdfObst1)
        env.add_obstacle(dynamicSphereObst3)
    if goal:
        from examples.scene_objects.goal import splineGoal

        env.add_goal(splineGoal)
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_point_robot(render=True)
