import argparse
import gym
import numpy as np
from itertools import count
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=True, action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--environment', type=str, default='Pong-v0',
                    help='Select prefered test environment')
args = parser.parse_args()


env = gym.make('Pong-v0')
env.seed(args.seed)  # Set random seed for the enviro
torch.manual_seed(args.seed)  # Set random seed for pytorch
action_space = env.action_space.n  # possible actions
state_space = env.observation_space.shape  # Size of the observation space
size_in = 80 * 80

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(size_in, 200)
        self.affine2 = nn.Linear(200, action_space)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)  # look into decaying


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)  # Convert from numpy
    probs = policy(Variable(state))  # Make Variable enables gradients
    m = Categorical(probs)  # categorical distribution based on probablities
    action = m.sample()  # Sample from distribuiton
    policy.saved_log_probs.append(m.log_prob(action))  # Save Log probablities
    return action.data[0]  # Return action sampled


def finish_episode():
    R = 0  # Discounted future rewards, set to 0 each time called
    policy_loss = []  # new loss calc for each game
    rewards = []  # New discounted rewards calc for each game
    for r in policy.rewards[::-1]:  # Reverse loop through actual rewards
        R = r + args.gamma * R  # Create R based on discounted future reward est
        rewards.insert(0, R)  # Insert R at the front for rewards
    rewards = torch.Tensor(rewards)  # Convert to pytorch tensor
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)  # Standardized to unit normal
    for log_prob, reward in zip(policy.saved_log_probs, rewards):  # loop through log_prob & rewards
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()  # Zero gradients before backward pass
    policy_loss = torch.cat(policy_loss).sum()  # Sum loss over last batch
    policy_loss.backward()  # Calculate gradients and pass back through graphs
    optimizer.step()  # Make changes to weights
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = deque(maxlen=10)
    prev_x = None  # used in computing the difference frame
    for i_episode in count(1):
        reward_sum = 0
        done = False
        state = env.reset()
        while not done:  # Don't infinite loop while learning
            # preprocess the observation, set input to network to be difference image
            if args.render and i_episode % 20 == 0:
                env.render()
            cur_x = prepro(state)
            x = cur_x - prev_x if prev_x is not None else np.zeros(size_in)
            prev_x = cur_x
            action = select_action(x)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            reward_sum += reward
        running_reward.append(reward_sum)
        if i_episode % 1 == 0:
            finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode: {}\tScore: {}\tAverage score: {}'.format(
                i_episode, reward_sum, np.mean(running_reward)))


if __name__ == '__main__':
    main()
