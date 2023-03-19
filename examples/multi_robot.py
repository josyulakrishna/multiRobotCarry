import gym
import numpy as np
from urdfenvs.robots.tiago import TiagoRobot
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.robots.prius import Prius
# import sys
# sys.path.append("/home/josyula/Programs/MAS_Project/gym_envs_urdf/")
import pybullet
def run_multi_robot(n_steps=1000, render=False, obstacles=False, goal=False):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
        # GenericUrdfReacher(urdf="ur5.urdf", mode="acc"),
        # GenericUrdfReacher(urdf="ur5.urdf", mode="acc"),
        # TiagoRobot(mode="vel"),
        # Prius(mode="vel")
    ]

    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    n = env.n()
    action = np.ones(n) * -0.2
    pos0 = np.zeros(n)
    pos0[1] = -0.0
    base_pos = np.array([
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        ])
    ob = env.reset(mount_positions=base_pos)
    print(f"Initial observation : {ob}")
    if goal:
        from examples.scene_objects.goal import dynamicGoal
        env.add_goal(dynamicGoal)

    # if obstacles:
    #     from examples.scene_objects.obstacles import dynamicSphereObst2
    #     env.add_obstacle(dynamicSphereObst2)
    pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    print("Starting episode")
    history = []
    i=0
    for _ in range(1000):
        ob, _, _, _ = env.step(action)
        history.append(ob)
        i+=1
        if i%100==0:
            env.reset(mount_positions=base_pos)
    env.close()
    return history

    #add load onto robots

if __name__ == "__main__":
    run_multi_robot(render=True, obstacles=True, goal=True)
