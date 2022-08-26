import torch
import torch.nn as nn


def guassian_kernel(source, target, kernel_mul, kernel_num, fix_sigma):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i)
                      for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def linear_mmd2(f_of_X, f_of_Y):
    delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
    loss = delta.dot(delta.T)
    return loss


class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()

    def forward(self, f_s, f_t):
        n = int(f_s.size()[0])
        kernels = guassian_kernel(
            f_s, f_t, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = torch.mean(kernels[:n, :n])
        YY = torch.mean(kernels[n:, n:])
        XY = torch.mean(kernels[:n, n:])
        YX = torch.mean(kernels[n:, :n])
        loss = torch.mean(XX + YY - XY - YX)
        return loss
