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
            nn.Linear(len_states, 256),
            nn.ReLU(),

            nn.Linear(256, 100),
            nn.ReLU(),

            nn.Linear(100, self.num_actions)
        )
        initialize_weights(self.model)

    def forward(self, x):
        action_value = self.model(x)
        return action_value


class DQN(nn.Module):
    def __init__(self, len_states: int, batch_size: int):
        super(DQN, self).__init__()
        self.epsilon = 0.9
        self.gamma = 0.9
        self.num_actions = 2
        self.batch_size = batch_size
        self.memory_counter = 0
        self.memory_capacity = 300
        self.memory = deque(maxlen=self.memory_capacity)
        self.eval_net, self.target_net = Net(len_states, self.num_actions), Net(len_states, self.num_actions)

    def choose_action(self, x):
        x = torch.unsqueeze(x, 0)
        if np.random.uniform() < self.epsilon:
            action_value = self.eval_net.forward(x)
            action = np.array(torch.max(action_value, 1)[1].tolist())
            action = action[0]
        else:
            action = np.random.randint(0, self.num_actions)
        return action

    def store_transition(self, state, action, reward, state_):
        # state:选择的样本序列
        transition = (state, action, reward, state_)
        self.memory.append(transition)
        self.memory_counter += 1

    def forward(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = [self.memory[i] for i in sample_index]
        b_states = torch.stack([i[0] for i in b_memory], dim=0)
        # print(b_states, b_states.size())
        b_actions = torch.stack([i[1] for i in b_memory], dim=0).unsqueeze(1)
        b_actions = b_actions.long()
        # print(b_actions, b_actions.size())
        b_rewards = torch.stack([i[2] for i in b_memory], dim=0).unsqueeze(1)
        print(b_rewards, b_rewards.size(), b_rewards.is_cuda)
        b_states_ = torch.stack([i[3] for i in b_memory], dim=0)
        # print(b_states_, b_states_.size())
        # print(torch.masked_select(b_states, b_states != 0).size())
        # print(torch.masked_select(b_states_, b_states_ != 0).size())

        q_eval = self.eval_net(b_states)
        print(q_eval.size())
        q_eval = q_eval.gather(1, b_actions)
        q_next = self.target_net(b_states_).detach()
        q_target = b_rewards + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        return q_eval, q_target
