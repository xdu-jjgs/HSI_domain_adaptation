# import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discrepancy(nn.Module):
    def __init__(self):
        super(Discrepancy, self).__init__()
        # self.eps = eps

    @property
    def num_keys(self):
        return 2

    def forward(self, p1, p2):
        # target: tensor(b, c, h, w); prediction: same as target
        if not p1.shape == p2.shape:
            raise ValueError('Input images must have the same dimensions.')
        loss = torch.mean(torch.abs(F.softmax(p1) - F.softmax(p2)))
        return loss


class ExpMinusMSE(nn.Module):
    def __init__(self):
        super(ExpMinusMSE, self).__init__()
        self.reduction = 'mean'
        # self.eps = eps

    def forward(self, prediction, target):
        # target: tensor(b, c, h, w); prediction: same as target
        if not target.shape == prediction.shape:
            raise ValueError('Input images must have the same dimensions.')
        assert target.ndim == 4 and target.shape[1] > 1, "image n_channels should be greater than 1"
        mse = F.mse_loss(prediction, target, reduction=self.reduction)
        loss = torch.exp(-mse)
        return loss


class ExpMSE(nn.Module):
    def __init__(self):
        super(ExpMSE, self).__init__()
        self.reduction = 'mean'
        # self.eps = eps

    def forward(self, prediction, target):
        # target: tensor(b, c, h, w); prediction: same as target
        if not target.shape == prediction.shape:
            raise ValueError('Input images must have the same dimensions.')
        assert target.ndim == 4 and target.shape[1] > 1, "image n_channels should be greater than 1"
        mse = F.mse_loss(prediction, target, reduction=self.reduction)
        loss = torch.exp(mse)
        return loss


# class FocalLoss(nn.Module):
#     r"""
#         only need to sent the class_num when using!!!!!!!!
#
#         This criterion is a implemenation of Focal Loss, which is proposed in
#         Focal Loss for Dense Object Detection.
#
#             Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
#
#         The losses are averaged across observations for each minibatch.
#
#         Args:
#             alpha(1D Tensor, Variable) : the scalar factor for this criterion
#             gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
#                                    putting more focus on hard, misclassiﬁed examples
#             size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                                 However, if the field size_average is set to False, the losses are
#                                 instead summed for each minibatch.
#
#
#     """
#
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average
#
#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs)
#
#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         # print(class_mask)
#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.to(device)
#         alpha = self.alpha[ids.data.view(-1)]
#
#         probs = (P * class_mask).sum(1).view(-1, 1)
#
#         log_p = probs.log()
#         # print('probs size= {}'.format(probs.size()))
#         # print(probs)
#
#         batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
#         # print('-----bacth_loss------')
#         # print(batch_loss)
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss


# if __name__ == "__main__":
#     loss = SAMLoss()
#     a = torch.rand(1, 102, 160, 160).cuda()
#     b = a
#     # b = torch.rand(1, 102, 160, 160).cuda()
#     print(loss(a,b))