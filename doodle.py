import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
import matplotlib.pyplot as plt
# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch Advantage Actor critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='NU',
                    help='num_epsiodes (default: 1000)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('MountainCar-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
num_inputs = 2
epsilon = 0.99
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def epsilon_value(epsilon):
    eps = 0.99*epsilon
    return eps


class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()
        self.Linear1 = nn.Linear(num_inputs, 64)
        nn.init.xavier_uniform(self.Linear1.weight)
        self.Linear2 = nn.Linear(64, 128)
        nn.init.xavier_uniform(self.Linear2.weight)
        self.Linear3 = nn.Linear(128, 64)
        nn.init.xavier_uniform(self.Linear3.weight)
        num_actions = env.action_space.n

        self.actor_head = nn.Linear(64, num_actions)
        self.critic_head = nn.Linear(64, 1)
        nn.init.xavier_uniform(self.critic_head.weight)
        self.action_history = []
        self.rewards_achieved = []

    def forward(self, state_inputs):
        x = F.relu(self.Linear1(state_inputs))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))
        return self.critic_head(x), x

    def act(self, state_inputs, eps):
        value, x = self(state_inputs)
        x = F.softmax(self.actor_head(x), dim=-1)
        m = Categorical(x)
        e_greedy = random.random()
        if e_greedy > eps:
            action = m.sample()
        else:
            action = m.sample_n(3)
            pick = random.randint(-1, 2)
            action = action[pick]
        return value, action, m.log_prob(action)


model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=0.002)


def perform_updates():
    '''
    Updating the ActorCritic network params
    '''
    r = 0
    saved_actions = model.action_history
    returns = []
    rewards = model.rewards_achieved
    policy_losses = []
    critic_losses = []

    for i in rewards:
        r = args.gamma*r + i
        returns.append(r)
    returns = torch.tensor(returns)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculating policy loss
        policy_losses.append(-log_prob * advantage)

        # calculating value loss
        critic_losses.append(F.mse_loss(value, torch.tensor([R])))
    optimizer.zero_grad()

    # Finding cumulative loss
    loss = torch.stack(policy_losses).sum() + torch.stack(critic_losses).sum()
    loss.backward()
    optimizer.step()
    # Action history and rewards cleared for next episode
    del model.rewards_achieved[:]
    del model.action_history[:]
    return loss.item()

def main():
    eps = epsilon_value(epsilon)
    losses= []
    counters = []
    plot_rewards = []

    for i_episode in range(0, args.num_episodes):
        counter = 0
        state = env.reset()
        ep_reward = 0
        done = False

        while not done:

            # unrolling state and getting action from the nn output
            state = torch.from_numpy(state).float()
            value, action, ac_log_prob = model.act(state, eps)
            model.action_history.append(SavedAction(ac_log_prob, value))
            # Agent takes the action
            state, reward, done, _ = env.step(action.item())

            '''if i_episode % 50 == 0: #uncomment if you want to see the training happen live
             env.render()'''

            model.rewards_achieved.append(reward)
            ep_reward += reward
            counter += 1
            if counter % 5 == 0:
                ''' performing backprop at every 5 time steps to avoid 
                    highly correlational states (sampling traces from episode) '''
                loss = perform_updates()
            eps = epsilon_value(eps)
            # decaying epsilon at the rate of 0.99 after each episode

        # saving the losses, num of timesteps before convergence and rewards
        if i_episode % args.log_interval == 0:
            losses.append(loss)
            counters.append(counter)
            plot_rewards.append(ep_reward)

    # plotting loss
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.savefig('loss1.png')

    # plotting number of timesteps elapsed before convergence
    plt.clf()
    plt.xlabel('Episodes')
    plt.ylabel('timesteps')
    plt.plot(counters)
    plt.savefig('timestep.png')
    # plotting total rewards achieved during all episodes
    plt.clf()
    plt.xlabel('Episodes')
    plt.ylabel('rewards')
    plt.plot(plot_rewards)
    plt.savefig('rewards.png')


if __name__ == '__main__':
    main()