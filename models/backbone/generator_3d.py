import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            # batch_size x in_channels x 64 x 64
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
            # batch_size x filters x 64 x 64
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, inputs):
        output = self.main(inputs)
        output += self.shortcut(inputs)
        return output


class GeneratorImage_3D(nn.Module):
    def __init__(self, data_c=103, cn=9):
        super(GeneratorImage_3D, self).__init__()
        self.data_c = data_c
        self.cn = cn
        self.in_c = data_c + cn
        # in_c*1*1 ==> (in_c*4*2*2)*1*1
        self.fc = nn.Linear(in_features=self.in_c, out_features=self.data_c*2*2)
        # self.reshape = torch.reshape([-1, in_c*4, 2, 2])  # (in_c*4*2*2)*1*1 ==> (in_c*4)*2*2
        # h_out = (h_in - 1) * stride + kernl_size - 2*padding + output_padding
        self.gnet_noise = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=1, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                               padding=(0, 1, 1), output_padding=(0, 1, 1)),  # 4*4
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                               padding=(1, 1, 1), output_padding=(0, 0, 0)),  # 7*7
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(5, 3, 3), stride=(1, 2, 2),
                               padding=(2, 1, 1), output_padding=(0, 1, 1)),  # 14*14
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=(7, 3, 3), stride=(1, 2, 2),
                               padding=(3, 1, 1), output_padding=(0, 0, 0)),  # 27*27
            nn.BatchNorm3d(1),
            nn.Tanh()
        )
        blocks = []
        for block in range(4):
            blocks.append(ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.gnet_image = nn.Sequential(
            nn.Conv2d(self.data_c, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            *blocks,
            nn.Conv2d(64, self.data_c, 3, 1, 1, bias=True),
            nn.Tanh()
        )
        self.last_activ = nn.Tanh()

    def forward(self, z, label, x):
        data_c = self.data_c
        # print(x.shape, label.shape)
        z = torch.cat((z, label), dim=1)
        # print(x.shape)
        z = self.fc(z)
        z = z.reshape([-1, 1, data_c, 2, 2])
        z = self.gnet_noise(z)
        z = torch.squeeze(z, dim=1)
        x = x + z
        x = self.last_activ(x)
        # x = self.gnet_image(x)
        return x


class Generator_3D(nn.Module):
    def __init__(self, data_c=103, cn=9, label_embedding=False):
        super(Generator_3D, self).__init__()
        # self.data_c = data_c
        self.in_c = data_c + cn
        self.data_c = data_c
        # in_c*1*1 ==> (in_c*4*2*2)*1*1
        self.label_embedding = label_embedding
        self.fc = nn.Linear(in_features=self.in_c, out_features=data_c * 2 * 2)
        if self.label_embedding:
            self.in_c = data_c + 64
            self.embedding = nn.Embedding(cn, 64)
            self.fc = nn.Linear(in_features=self.in_c, out_features=data_c * 2 * 2)
        # self.reshape = torch.reshape([-1, in_c*4, 2, 2])  # (in_c*4*2*2)*1*1 ==> (in_c*4)*2*2
        # h_out = (h_in - 1) * stride + kernl_size - 2*padding + output_padding
        self.gnet_noise = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=1, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                               padding=(0, 1, 1), output_padding=(0, 1, 1)),    # 4*4
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                               padding=(1, 1, 1), output_padding=(0, 0, 0)),  # 7*7
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(5, 3, 3), stride=(1, 2, 2),
                               padding=(2, 1, 1), output_padding=(0, 1, 1)),  # 14*14
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=(7, 3, 3), stride=(1, 2, 2),
                               padding=(3, 1, 1), output_padding=(0, 0, 0)),  # 27*27
            nn.BatchNorm3d(1),
            nn.Tanh()
        )

    def forward(self, z, label):
        data_c = self.data_c
        # print(x.shape, label.shape)
        z = torch.cat((z, label), dim=1)
        # print(x.shape)
        z = self.fc(z)
        z = z.reshape([-1, 1, data_c, 2, 2])
        z = self.gnet_noise(z)
        z = torch.squeeze(z, dim=1)
        return z
