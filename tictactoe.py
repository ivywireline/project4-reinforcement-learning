from __future__ import print_function, division
from collections import defaultdict, OrderedDict
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done

class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.softmax(x, dim=1)

def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr)
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    # TODO
    result = []
    for index in range(len(rewards)):
        sum_returns = 0
        power = 0
        for i in range(index, len(rewards)):
            sum_returns = sum_returns + ((gamma ** power) * rewards[i])
            power = power + 1
        result.append(sum_returns)
    return result

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 0, # TODO
            Environment.STATUS_INVALID_MOVE: -99,
            Environment.STATUS_WIN         : 2,
            Environment.STATUS_TIE         : 1,
            Environment.STATUS_LOSE        : -1
    }[status]

def train(policy, env, gamma=1.0, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    performance_data = OrderedDict()
    invalid_moves_episode = []

    for i_episode in count(1):
        num_invalid_moves = 0
        if i_episode > 50000:
            break
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            if status == Environment.STATUS_INVALID_MOVE:
                num_invalid_moves += 1
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        invalid_moves_episode.append(num_invalid_moves)

        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            performance_data[i_episode] = running_reward / log_interval
            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return {
        'invalid_moves_episode': invalid_moves_episode,
        'performance_data': performance_data,
    }


def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)


def plot_learning_curve(
    learning_data,
    filename="Part5LearningCurve",
    plot_label="Average return",
    xlabel="Episodes",
    ylabel="Average Return",
    title="Learning Curves of Average Return vs Episodes",
):

    x_axis = learning_data.keys()
    y_axis = [learning_data[i] for i in x_axis]

    fig = plt.figure()
    plt.plot(x_axis, y_axis, label=plot_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")

    if filename:
        plt.savefig(filename)


def plot_invalid_move_performance(invalid_move_list, filename="Part5cLearningCurve"):
    labels = range(1000, 51000, 5000)
    xs = np.arange(len(labels))
    ys = invalid_move_list[::5]

    width = 1

    fig = plt.figure()
    #plt.plot(x_axis, y_axis, label="Num of Invalid Moves")
    plt.bar(xs, ys, width, align='center', label="Num of Invalid Moves")
    plt.xticks(xs, labels)
    plt.yticks(ys)

    plt.xlabel("Episodes")
    plt.ylabel("Number of Invalid Moves")
    plt.title("Number of Invalid Moves per 5000 Episodes")
    plt.legend(loc="best")

    if filename:
        plt.savefig(filename)


def part5b():
    new_policy = Policy(hidden_size=128)
    new_env = Environment()
    new_train_summary = train(new_policy, new_env, gamma=0.9)
    plot_learning_curve(new_train_summary['performance_data'], "Part5BLearningCurve_128")


    new_policy = Policy(hidden_size=32)
    new_env = Environment()
    new_train_summary = train(new_policy, new_env, gamma=0.9)
    plot_learning_curve(new_train_summary['performance_data'], "Part5BLearningCurve_32")

    new_policy = Policy(hidden_size=80)
    new_env = Environment()
    new_train_summary = train(new_policy, new_env, gamma=0.9)
    plot_learning_curve(new_train_summary['performance_data'], "Part5BLearningCurve_80")


def part5d(policy, env):
    wins = 0
    ties = 0
    losses = 0
    print()
    for i in range(100):
        # show_games = i % 20 == 0
        show_games = True
        if show_games:
            print("Game #{}:".format(i+1))
        w, t, l = play_games_against_random(policy, env, num_games=1, show_games=show_games)
        if show_games:
            print()
        wins += w
        ties += t
        losses += l
    return wins, ties, losses


def part6():
    # x axis
    episode_lst = [i for i in range(1000, 51000, 1000)]
    y_axis = []
    for i in episode_lst:
        policy = Policy(hidden_size=80)
        env = Environment()
        load_weights(policy, i)
        y_axis.append(play_games_against_random(policy, env))
    plot_learning_curve_part6(episode_lst, y_axis)


def part7(policy, env):
    ep = 50000
    load_weights(policy, ep)
    print (first_move_distr(policy, env)[0][0])

    # Explore how first move distribution varies throughout training
    # x axis
    episode_lst = [i for i in range(1000, 51000, 1000)]
    y_axis = []
    for i in episode_lst:
        load_weights(policy, i)
        y_axis.append(first_move_distr(policy, env)[0])
    plot_learning_curve_part7(episode_lst, y_axis)



def plot_learning_curve_part6(
    x_axis,
    y_axis,
    filename="Part6LearningCurve",
):

    y_axis_win_rate = [item[0] for item in y_axis]
    y_axis_tie_rate = [item[1] for item in y_axis]
    y_axis_lose_rate = [item[2] for item in y_axis]

    fig = plt.figure()
    plt.plot(x_axis, y_axis_win_rate, 'r-', label="Win Rate")
    plt.plot(x_axis, y_axis_tie_rate, 'y-', label="Tie Rate")
    plt.plot(x_axis, y_axis_lose_rate, label="Lose Rate")

    plt.xlabel("Episodes")
    plt.ylabel("Win/Lose/Tie Rates")
    plt.title("Win Rate Over Episodes")
    plt.legend(loc="best")

    if filename:
        plt.savefig(filename)


def plot_learning_curve_part7(
    x_axis,
    y_axis,
    filename="Part7LearningCurve",
):

    y_axis_column0 = [item[0] for item in y_axis]
    y_axis_column1 = [item[1] for item in y_axis]
    y_axis_column2 = [item[2] for item in y_axis]
    y_axis_column3 = [item[3] for item in y_axis]
    y_axis_column4 = [item[4] for item in y_axis]
    y_axis_column5 = [item[5] for item in y_axis]
    y_axis_column6 = [item[6] for item in y_axis]
    y_axis_column7 = [item[7] for item in y_axis]
    y_axis_column8 = [item[8] for item in y_axis]

    fig = plt.figure()
    plt.plot(x_axis, y_axis_column0, label="Column 0")
    plt.plot(x_axis, y_axis_column1, label="Column 1")
    plt.plot(x_axis, y_axis_column2, label="Column 2")
    plt.plot(x_axis, y_axis_column3, label="Column 3")
    plt.plot(x_axis, y_axis_column4, label="Column 4")
    plt.plot(x_axis, y_axis_column5, label="Column 5")
    plt.plot(x_axis, y_axis_column6, label="Column 6")
    plt.plot(x_axis, y_axis_column7, label="Column 7")
    plt.plot(x_axis, y_axis_column8, label="Column 8")

    plt.xlabel("Episodes")
    plt.ylabel("Probability of Making the First Move")
    plt.title("First Move Distribution over Episodes")
    plt.legend(loc="best")

    if filename:
        plt.savefig(filename)


def play_games_against_random(policy, env, num_games=100, show_games=False):
    wins = 0
    ties = 0
    losses = 0
    for i in range(num_games):
        # print("Playing game", i+1)
        done = False
        num_invalid_moves = 0
        state = env.reset()
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            if status == Environment.STATUS_INVALID_MOVE:
                num_invalid_moves += 1
            if show_games:
                env.render()

        if status == Environment.STATUS_WIN:
            wins += 1
        elif status == Environment.STATUS_TIE:
            ties += 1
        elif status == Environment.STATUS_LOSE:
            losses += 1
        else:
            print("Something has gone terribly wrong")

        if show_games:
            if status == Environment.STATUS_WIN:
                print("Win!")
            elif status == Environment.STATUS_TIE:
                print("Tie")
            elif status == Environment.STATUS_LOSE:
                print("Loss")
    return wins, ties, losses


if __name__ == '__main__':
    import sys
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    policy = Policy(hidden_size=80)
    env = Environment()

    if len(sys.argv) == 1:
        # `python tictactoe.py` to train the agent
        train_summary = train(policy, env, gamma=0.9)
        plot_learning_curve(train_summary['performance_data'])
        invalid_moves_per_1k = [sum(train_summary['invalid_moves_episode'][i * 1000:(i + 1000) * 1000]) for i in range(int(50000 / 1000))]

        for i, val in enumerate(invalid_moves_per_1k):
            print("{}\t{}".format((i+1)*1000, val / 1000))

        plot_invalid_move_performance(invalid_moves_per_1k)


        # Uncomment this like to see the tuning of the number of hidden units hyperparameter
        # in part 5 b)
        # part5b()
    elif len(sys.argv) == 2:
        # `python tictactoe.py <ep>` to print the first move distribution
        # using weightt checkpoint at episode int(<ep>)
        ep = int(sys.argv[1])
        load_weights(policy, ep)
        print(first_move_distr(policy, env))

        # Part5d
        wins, ties, losses = part5d(policy, env)
        print("Wins: {}%\tTies: {}\tLosses: {}".format(wins, ties, losses))
    else:
        # `python tictactoe.py -f "part6 and part7"`
        part6()
        part7(policy, env)
