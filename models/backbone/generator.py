import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, data_c=103, cn=9):
        super(Generator, self).__init__()
        self.data_c = data_c
        self.cn = cn
        self.in_c = data_c + cn
        # in_c*1*1 ==> (in_c*4*2*2)*1*1
        self.fc = nn.Linear(in_features=self.in_c, out_features=self.in_c*4*2*2)
        # self.reshape = torch.reshape([-1, in_c*4, 2, 2])  # (in_c*4*2*2)*1*1 ==> (in_c*4)*2*2
        # h_out = (h_in - 1) * stride + kernl_size - 2*padding + output_padding
        self.gnet = nn.Sequential(
            nn.BatchNorm2d(self.in_c*4),
            nn.ReLU(),
            nn.ConvTranspose2d(self.in_c*4, self.data_c*4, 3, 2, 1, 1),  # 4*4
            nn.BatchNorm2d(self.data_c*4),
            nn.ReLU(),
            nn.ConvTranspose2d(self.data_c*4, self.data_c*2, 3, 2, 1),  # 7*7
            nn.BatchNorm2d(self.data_c*2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.data_c*2, self.data_c, 3, 2, 1, 1),  # 14*14
            nn.BatchNorm2d(self.data_c),
            nn.ReLU(),
            nn.ConvTranspose2d(self.data_c, self.data_c, 3, 2, 1),  # 27*27
            nn.BatchNorm2d(self.data_c),
            nn.Tanh()
        )

    def forward(self, x, label):
        in_c = self.in_c
        # print(x.shape, label.shape)
        x = torch.cat((x, label), dim=1)
        # print(x.shape)
        x = self.fc(x)
        x = x.reshape([-1, in_c*4, 2, 2])
        x = self.gnet(x)
        return x
