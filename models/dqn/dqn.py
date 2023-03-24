import torch
import numpy as np
import torch.nn as nn

from collections import deque
from models.utils.init import initialize_weights


class Net(nn.Module):
    def __init__(self, len_states: int, num_actions: int):
        super(Net, self).__init__()
        # 只输入序列
        # 可以单智能体/多智能体
        self.num_actions = num_actions
        self.model = nn.Sequential(
            nn.Linear(len_states, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 32),
            nn.LeakyReLU(),

            nn.Linear(32, self.num_actions)
        )
        initialize_weights(self.model)

    def forward(self, x):
        action_value = self.model(x)
        return action_value


class DQN(nn.Module):
    # TODO: change memory_capacity init
    def __init__(self, len_states: int, num_actions: int, batch_size:int, memory_capacity: int = 2000, step_observe: int = 100):
        super(DQN, self).__init__()
        self.batch_size = batch_size
        self.epsilon = 0.9
        self.epsilon_min = 0.
        self.epsilon_max = 0.9
        self.gamma = 0.9
        self.num_actions = num_actions
        self.step = 0
        self.step_observe = step_observe
        self.explore = 10000
        self.memory_capacity = memory_capacity
        self.memory = deque(maxlen=self.memory_capacity)
        self.current_state = None
        self.eval_net, self.target_net = Net(len_states, self.num_actions), Net(len_states, self.num_actions)

    def choose_action(self, selected_num):
        # x = torch.unsqueeze(x, 0)
        action = torch.zeros(self.num_actions)
        if np.random.random() <= self.epsilon:
            action_index = np.random.randint(self.CANDIDATE_NUM - selected_num)
            action[action_index] = 1
        else:
            action_value = self.eval_net.forward(self.current_state)
            action_index = np.argmax(action_value[0, 0:(self.CANDIDATE_NUM - selected_num)])
            action[action_index] = 1
        if self.epsilon > self.epsilon_ and self.step > self.step_observe:
            self.epsilon -= (self.epsilon_max - self.epsilon_min) / self.explore
        return action, action_index

    def store_transition(self, action, reward, next_state, num_select, terminal, iteration):
        # state:选择的样本序列
        transition = (self.current_state, action, reward, next_state, num_select, terminal, iteration)
        self.memory.append(transition)
        if self.step > self.step_observe and terminal == 1:
            self.train_net()
        self.step += 1
        self.current_state = next_state

    def train_net(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = [self.memory[i] for i in sample_index]
        b_states = torch.stack([i[0] for i in b_memory], dim=0)
        # print(b_states, b_states.size())
        b_actions = torch.stack([i[1] for i in b_memory], dim=0).unsqueeze(1)
        b_actions = b_actions.long()
        # print(b_actions, b_actions.size())
        b_rewards = torch.stack([i[2] for i in b_memory], dim=0).unsqueeze(1)
        # print(b_rewards, b_rewards.size())
        b_new_states = torch.stack([i[3] for i in b_memory], dim=0)
        b_num_select = torch.stack([i[4] for i in b_memory], dim=0)
        # print(b_new_states, b_new_states.size())
        # print(torch.masked_select(b_states, b_states != 0).size())
        # print(torch.masked_select(b_new_states, b_new_states != 0).size())

        q_eval = self.eval_net(b_num_select)
        q_eval = q_eval.gather(1, b_actions)
        q_next = self.target_net(b_new_states).detach()
        q_target = b_rewards + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        return q_eval, q_target


def get_next_state(state, num_select):
    feature_dim = state.size()[0]
    zero_padding = torch.zeros([num_select, feature_dim])
    new_state = torch.cat([state, zero_padding])
    return new_state
