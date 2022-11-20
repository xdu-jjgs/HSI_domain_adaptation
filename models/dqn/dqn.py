import torch
import numpy as np
import torch.nn as nn

from collections import deque
from models.utils.init import initialize_weights


class Net(nn.Module):
    def __init__(self, len_states: int):
        super(Net, self).__init__()
        # 只输入序列
        # 可以单智能体/多智能体
        self.actions = 2
        self.model = nn.Sequential(
            nn.Linear(len_states, 256),
            nn.ReLU(),

            nn.Linear(256, 100),
            nn.ReLU(),

            nn.Linear(100, self.actions)
        )
        initialize_weights(self.model)

    def forward(self, x):
        action_value = self.model(x)
        return action_value


class DQN(nn.Module):
    def __init__(self, len_states: int, batch_size: int):
        super(DQN).__init__()
        self.epsilon = 0.9
        self.gamma = 0.9
        self.batch_size = batch_size
        self.memory_counter = 0
        self.memory_capacity = 300
        self.memory = deque(maxlen=self.memory_capacity)
        self.eval_net, self.target_net = Net(len_states), Net(len_states)

    def choose_action(self, x):
        # x = torch.unsqueeze(x, 0)
        if np.random.uniform() < self.epsilon:
            action_value = self.eval_net.forward(x)
            action = np.array(torch.max(action_value, 1)[1].tolist())
            action = action[0]
        else:
            action = np.random.randint(0, self.N_ACTIONS)
        return action

    def store_transition(self, state, action, reward, state_):
        # state:选择的样本序列
        transition = (state, action, reward, state_)
        self.memory.append(transition)
        self.memory_counter += 1

    def forward(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_states = torch.tensor([i[0] for i in b_memory])
        b_actions = torch.tensor([i[1] for i in b_memory])
        b_rewards = torch.FloatTensor([i[2] for i in b_memory])
        b_states_ = torch.tensor([i[3] for i in b_memory])

        q_eval = self.eval_net(b_states).gather(1, b_actions)
        q_next = self.target_net(b_states_).detach()
        q_target = b_rewards + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        return q_eval, q_target
