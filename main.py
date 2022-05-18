import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import time
from RL import Actor_Critic
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
from env import RAEnv
import pygame
from get_dynamics import get_dynamics
from ECBF import ECBF_control, discrete_ECBF, continuous_ECBF_version1

pygame.init()
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="RAEnv-v0",
                    help='Environment (default: RAEnv-v0)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 50 episode (default: True)')
parser.add_argument('--modular', action="store_true",
                    help='Uses two agents, one for each task (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.003, metavar='G',
                    help='learning rate (default: 0.003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)


# Agent

print("modular:", args.modular)

print('cuda', args.cuda)

print(env.observation_space.shape[0])

agent_1 = Actor_Critic(env.observation_space.shape[0], env.action_space, args)
agent_1.total_numsteps = 0
if args.modular:
    agent_2 = Actor_Critic(env.observation_space.shape[0], env.action_space, args)
    agent_2.total_numsteps = 0

#TesnorboardX
writer_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

writer_1 = SummaryWriter(logdir='runs/{}_Actor_Critic_1_{}_{}_{}'.format(writer_datetime, args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

if args.modular:
    writer_2 = SummaryWriter(logdir='runs/{}_Actor_Critic_2_{}_{}_{}'.format(writer_datetime, args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory_1 = ReplayMemory(args.replay_size,args.seed)
if args.modular:
    memory_2 = ReplayMemory(args.replay_size,args.seed)

# Training Loop
total_numsteps = 0
updates_1 = 0
if args.modular:
    updates_2 = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    i_episode_rewards = []

    done = False
    state = env.reset()
    while (env.state[0] > 0.8 or env.state[0] < -0.8):
        state = env.reset()
    while not done:
        if not args.modular or state[-2] == 0:
            agent = agent_1
            memory = memory_1
            writer = writer_1
            updates = updates_1
        else:
            agent = agent_2
            memory = memory_2
            writer = writer_2
            updates = updates_2

        if args.start_steps > agent.total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                updates += 1
        action_RL = action
        f, g, x = get_dynamics(state, action_RL)
        u_cbf, slack = ECBF_control(x, action_RL, f, g)

        action = action_RL + u_cbf
        next_state, reward, done, _ = env.step(action) # Step

        if abs(u_cbf) > 0.1:
            reward = -abs(float(u_cbf) + float(slack))
            action = action_RL

        episode_steps += 1
        agent.total_numsteps += 1
        episode_reward += reward


        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        i_episode_rewards.append(reward)
        angle_step = np.arctan2(next_state[1], next_state[0])
        i_episode_angle.append(abs(angle_step))

        state = next_state


    #if total_numsteps > args.num_steps:
    if i_episode > 10000:
        break

    writer_1.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

env.close()

