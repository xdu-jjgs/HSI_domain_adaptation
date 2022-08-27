import torch
import numpy as np
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


class BaseMMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=1):
        super(BaseMMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None


class MMDLoss(BaseMMDLoss):
    def __init__(self, kernel_mul=2.0, kernel_num=1):
        super(MMDLoss, self).__init__(kernel_mul, kernel_num)

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


class LocalMMDLoss(BaseMMDLoss):
    def __init__(self, kernel_mul=2.0, kernel_num=1):
        super(LocalMMDLoss, self).__init__(kernel_mul, kernel_num)

    def forward(self, f_s, f_t, label_s, y_t):
        batch_size = f_s.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(label_s, y_t)
        weight_ss = torch.from_numpy(weight_ss).cuda()  # B, B
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()

        kernels = self.guassian_kernel(f_s, f_t,
                                       kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        # Dynamic weighting
        lamb = self.lamb()
        self.step()
        loss = loss * lamb
        return loss

    def cal_weight(self, label_s, y_t):
        batch_size = label_s.size()[0]
        num_classes = y_t.size()[1]
        label_s = label_s.cpu().data.numpy()
        source_label_onehot = np.eye(num_classes)[label_s]  # one hot

        source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, num_classes)
        source_label_sum[source_label_sum == 0] = 100
        source_label_onehot = source_label_onehot / source_label_sum  # label ratio

        # Pseudo label
        target_label = y_t.cpu().data.max(1)[1].numpy()

        y_t = y_t.cpu().data.numpy()
        target_logits_sum = np.sum(y_t, axis=0).reshape(1, num_classes)
        target_logits_sum[target_logits_sum == 0] = 100
        y_t = y_t / target_logits_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(label_s)
        set_t = set(target_label)
        count = 0
        for i in range(num_classes):  # (B, C)
            if i in set_s and i in set_t:
                s_tvec = source_label_onehot[:, i].reshape(batch_size, -1)  # (B, 1)
                t_tvec = y_t[:, i].reshape(batch_size, -1)  # (B, 1)

                ss = np.dot(s_tvec, s_tvec.T)  # (B, B)
                weight_ss = weight_ss + ss
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st
                count += 1

        length = count
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
