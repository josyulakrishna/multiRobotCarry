import torch
import gym
import numpy as np
from torch.autograd import Variable as V
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from proportional import Experience
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from PPOERL import PPO
import pybullet as p

device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    # print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

###########PPO Params############
has_continuous_action_space = True  # continuous action space; else discrete

max_ep_len = 1200  # max timesteps in one episode
max_training_timesteps = int(1e5)  # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)  # save model frequency (in num timesteps)

action_std = 0.6  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(1e3)  # action_std decay frequency (in num timesteps)
#####################################################

## Note : print/log frequencies should be > than max_ep_len

################ PPO hyperparameters ################
update_timestep = max_ep_len * 2  # update policy every n timesteps
K_epochs = 30  # update policy for K epochs in one PPO update

eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network
################ PPO Params ################################

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


env, obs = make_env(render=False)
kill_env(env)
s_dim = 28 #env.observation_space.shape[0]
action_dim = 4
state_dim = 28
a_max = 0.5 #env.action_space.high[0]
a_min = -0.5 #env.action_space.low[0]
tau = 0.001
max_episodes = 50000
batch_size = 128
gamma = 0.99
EPS = 0.003
nIndivisual = 1
nElites = 1


class MemoryBuffer:

    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        s1_arr = np.float32([arr[3] for arr in batch])

        return s_arr, a_arr, r_arr, s1_arr

    def lens(self):
        return self.len

    def add(self, s, a, r, s1):
        """
        adds a particular transaction in the memory buffer
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param s1: next state
        :return:
        """
        transition = (s, a, r, s1)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, state_dim=state_dim, action_dim=action_dim):
        super(Actor, self).__init__()

        self.actor1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            # nn.Tanh()
        )
        self.fitness = -1000

    def forward(self, state):
        x = self.actor1(state)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim=state_dim, action_dim=action_dim):
        super(Critic, self).__init__()
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.fitness = -1000

    def forward(self, state):
        x = self.critic1(state)
        return x

def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim=action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

#
# class DDPG():
#     def __init__(self):
#         self.actor = Actor()
#         self.critic = Critic(s_dim, action_dim)
#         self.target_actor = Actor()
#         self.target_critic = Critic(s_dim, action_dim)
#         hard_update(self.target_actor, self.actor)
#         hard_update(self.target_critic, self.critic)
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), 1e-3)
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 1e-4)
#         self.noise = OrnsteinUhlenbeckActionNoise(action_dim)
#         self.buffer = Experience(int(1e6), batch_size, 0.5)
#         # if torch.cuda.is_available():
#         # self.actor.cuda()
#         # self.
#
#     def get_action(self, state, noise=True):
#         state = V(torch.Tensor(state))
#         self.actor.eval()
#         self.critic.eval()
#         self.actor.training = False
#         action = self.actor.forward(state).detach()
#         if noise:
#             new_action = action.data.numpy() + (self.noise.sample() * a_max)
#         else:
#             new_action = action.data.numpy()
#         self.actor.training = True
#         self.actor.train()
#         self.critic.train()
#         # print(new_action)
#         return new_action
#
#     def calculate_priority(self, data):
#
#         # s1 = V(torch.unsqueeze(torch.Tensor(data[0]),0))
#         # a1 = V(torch.unsqueeze(torch.Tensor(data[1]),0))
#         # r1 = V(torch.unsqueeze(torch.Tensor([data[2]]),0))
#         # s2 = V(torch.unsqueeze(torch.Tensor(data[3]),0))
#         # #data = V(torch.Tensor(data))
#         # s1 = data[:,0]
#         # a1 = data[:,1]
#         # r1 = data[:,2]
#         # s2 = data[:,3]
#         s1 = [arr[0] for arr in data]
#         a1 = [arr[1] for arr in data]
#         r1 = [arr[2] for arr in data]
#         s2 = [arr[3] for arr in data]
#         s1 = V(torch.Tensor(s1))
#         a1 = V(torch.Tensor(a1))
#         r1 = V(torch.Tensor(r1))
#         s2 = V(torch.Tensor(s2))
#
#         a2 = self.target_actor.forward(s2).detach()
#         next_q = torch.squeeze(self.target_critic.forward(s2, a2).detach())
#         y_expected = torch.squeeze(r1) + gamma * next_q
#         y_predicted = torch.squeeze(self.critic.forward(s1, a1).detach())
#
#         # a2 = self.target_actor.forward(s2).detach()
#         # next_q = torch.squeeze(self.target_critic.forward(s2,a2).detach())
#         # y_expected = r1 + gamma * next_q
#         # y_predicted = torch.squeeze(self.critic.forward(s1,a1).detach())
#         # print(y_predicted)
#         TD_error = y_expected - y_predicted
#         # TD_error = torch.squeeze(TD_error)
#
#         TD_error = abs(TD_error)
#
#         return TD_error
#
#     def add_to_buffer(self, data):
#         priority = self.calculate_priority([data])
#         self.buffer.add(data, priority.item())
#
#     def evaluate(self, trials=1):
#         total_steps = 0
#         for ep in range(trials):
#             env, obs = make_env(render=False)
#             print("episode", ep)
#             done = False
#             Reward = 0
#             ep_steps = 0
#             while not done:
#                 # if ep > 70:
#                 #     env.render()
#                 state = flatten_observation(obs)
#                 # state = np.float32(obs)
#                 action = self.get_action(state)
#                 action_new = np.clip(np.hstack((action.reshape(2, 2), np.zeros((2, 1)))).ravel(), -0.1, 0.1)
#                 new_obs, r, done, info = env.step(action_new)
#                 Reward += r[0]+r[1]
#                 total_steps += 1
#                 d = np.reshape(np.array(1.0 if done == True else 0.0, dtype=np.float32), (1, 1))
#                 if np.any(np.isnan(flatten_observation(new_obs))) or np.any(np.isnan(flatten_observation(obs))):
#                     done = True
#                 else:
#                     self.add_to_buffer([flatten_observation(obs), action, [r[0]+r[1]], flatten_observation(new_obs)])
#                     obs = new_obs
#                     self.experience_replay()
#                     ep_steps += 1
#                 if ep_steps > 1500:
#                     done = True
#             kill_env(env)
#             print("rl Reward:", Reward)
#             self.actor.fitness = Reward
#
#     def experience_replay(self):
#         if self.buffer.tree.filled_size() < batch_size:
#             return
#         out, we, idxes = self.buffer.select(batch_size)
#         s1 = [arr[0] for arr in out]
#         a1 = [arr[1] for arr in out]
#         r1 = [arr[2] for arr in out]
#         s2 = [arr[3] for arr in out]
#         s1 = V(torch.Tensor(s1))
#         a1 = V(torch.Tensor(a1))
#         r1 = V(torch.Tensor(r1))
#         s2 = V(torch.Tensor(s2))
#
#         # update critic
#         a2 = self.target_actor.forward(s2).detach()
#         next_q = torch.squeeze(self.target_critic.forward(s2, a2).detach())
#         y_expected = torch.squeeze(r1) + gamma * next_q
#         y_predicted = torch.squeeze(self.critic.forward(s1, a1))
#         # print(y_expected.shape)
#
#         loss_c = F.smooth_l1_loss(y_predicted, y_expected)
#         self.critic_optimizer.zero_grad()
#         loss_c.backward()
#         self.critic_optimizer.step()
#         # update actor
#         a = self.actor.forward(s1)
#         loss_a = -1 * torch.sum(self.critic.forward(s1, a))
#         self.actor_optimizer.zero_grad()
#         loss_a.backward()
#         self.actor_optimizer.step()
#         soft_update(self.target_actor, self.actor, tau)
#         soft_update(self.target_critic, self.critic, tau)
#
#         # update priority
#         priority = self.calculate_priority(out)
#         priority = [arr.item() for arr in priority]
#         self.buffer.priority_update(idxes, priority)
#

class ERL:
    def __init__(self):
        self.indivisual = []
        self.nIndivisual = nIndivisual
        self.nElites = nElites
        for i in range(self.nIndivisual):
            self.indivisual.append(Actor())
        self.ppo = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
        self.pMutation = 0.9
        self.omg = 1
        self.tornament_winners = []
        for i in range(self.nIndivisual - self.nElites):
            self.tornament_winners.append(Actor())

    def get_action(self, actor, state):
        state = V(torch.Tensor(state))
        actor.eval()

        actor.training = False
        action = actor.forward(state).detach()

        new_action = action.data.numpy()
        actor.training = True
        actor.train()

        # print(new_action)
        return new_action
    def evaluate(self, actor, times=1, episode=1):
        fitness = 0
        flag=False
        for i in range(times):
            env, obs = make_env(render=False)
            # p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
            #                     "/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/Evolutionary-Reinforcement-Learning/simple_no_pool/erl_passage.mp4")
            done = False
            total_steps = 0
            rf = 0
            robot_positions = np.zeros((2,2))
            goal_pos = np.array([8., 0.])
            while not done:
                state = flatten_observation(obs)
                action = self.get_action(actor, state)
                action_new = np.clip(np.hstack((action.reshape(2, 2), np.zeros((2, 1)))).ravel(), -0.1, 0.1)
                new_obs, r, done, info = env.step(action_new)
                # dist2goal, distbrobots, slopegoal, sloperobots = get_bc_features(new_obs)
                # if dist2goal > 0.2:
                #     rf = 1/(dist2goal)
                # # fitness += r[0] + r[1]+rf
                # for j, key in enumerate(new_obs.keys()):
                #     # get robot position
                #     robot_positions[j, :] = new_obs[key]['joint_state']['position'][:2]
                # area_tri = abs(np.cross(robot_positions[0, :] - robot_positions[1, :], goal_pos - robot_positions[0, :]))
                # area_tri = np.linalg.norm(area_tri) / 2
                # penalty = -(area_tri+0.001)*np.exp(-(dist2goal/8)+0.001)
                fitness += r[0] + r[1]
                total_steps += 1
                d = np.reshape(np.array(1.0 if done == True else 0.0, dtype=np.float32), (1, 1))
                if np.any(np.isnan(flatten_observation(new_obs))) or np.any(np.isnan(flatten_observation(obs))):
                    done = True
                else:
                    state1 = torch.FloatTensor(state).to(device)
                    action, action_logprob = self.ppo.policy_old.act(state1)

                    self.ppo.buffer.actions.append(action)
                    self.ppo.buffer.states.append(state1)
                    self.ppo.buffer.logprobs.append(action_logprob)
                    self.ppo.buffer.is_terminals.append(done)
                    self.ppo.buffer.rewards.append(r[0] + r[1])
                    obs = new_obs
                if total_steps > 3000:
                    done = True
                if info["goal_reached"]:
                    flag = True
            kill_env(env)
            print("Times:", times, "Fitness:", fitness)
            # wandb.log({"fittness": fitness, "goal": int(info["goal_reached"])})
        # p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
        return fitness / times , flag

    def rank(self):
        self.indivisual = sorted(self.indivisual, key=lambda x: x.fitness, reverse=True)

    def tornament(self, k):
        winners = []
        for i in range(self.nIndivisual - self.nElites):
            best = np.random.randint(0, self.nIndivisual - 1)
            for j in range(k):
                ind = np.random.randint(0, self.nIndivisual - 1)
                if self.indivisual[ind].fitness > self.indivisual[best].fitness:
                    best = ind
            winners.append(best)
            for i in range(len(winners)):
                hard_update(self.tornament_winners[i], self.indivisual[winners[i]])

    def mutation(self, actor):
        for target_param in actor.parameters():
            if target_param.data.size()[0] == 1:
                noised = torch.normal(target_param.data, torch.abs(target_param.data))
            else:
                noised = torch.normal(target_param.data, abs(target_param.data)+0.3)

            target_param.data.copy_(noised)

    def insert_rl(self):
        length = len(self.indivisual)
        self.rank()
        hard_update(self.indivisual[length - 1], self.ppo.policy.actor)

    def train(self):
        generation = 0
        for ep in range(2500):
            print("generation:", generation)
            for actor in range(len(self.indivisual)):
                flag= False
                self.indivisual[actor].fitness, flag = self.evaluate(self.indivisual[actor])
                if flag:
                    torch.save(self.indivisual[actor].state_dict(), 'Models/' + '1_actor_{0}.pt'.format(self.indivisual[actor].fitness))

            self.rank()
            elites = []
            for elite in range(self.nElites):
                a = self.indivisual[elite]
                elites.append(a)
            winners = self.tornament(1)
            for winner in self.tornament_winners:
                if np.random.random() < self.pMutation:
                    self.mutation(winner)
            for i in range(len(self.tornament_winners)):
                hard_update(self.indivisual[self.nElites + i], self.tornament_winners[i])
            if  len(self.ppo.buffer.actions)>2500 and generation%4==0:
                self.ppo.update()
            self.ppo.evaluate_fitness()

            if generation % self.omg == 0:
                self.insert_rl()
                self.save_models()
            # serializers.save_npz('my.model', self.indivisual[0])
            generation += 1

    def save_models(self):
        torch.save(self.indivisual[0].state_dict(), 'Models/' + '1_actor.pt')

    def load_models(self):
        self.indivisual[0].load_state_dict(torch.load('Models/' + 'best_sofar.pt'))


if __name__ == '__main__':
    # ddpg = DDPG()
    # ddpg.evaluate(2000)
    erl = ERL()
    erl.train()
    erl.save_models()
    # erl.load_models()
    # erl.evaluate(erl.indivisual[0],times=1,episode=80)