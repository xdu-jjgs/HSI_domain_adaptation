import torch

from criterions.base import TransferLoss


class CoralLoss(TransferLoss):
    def __init__(self):
        super(CoralLoss, self).__init__()

    def forward(self, f_s, f_t):
        d = f_s.data.shape[1]
        ns, nt = f_s.data.shape[0], f_t.data.shape[0]
        # source covariance
        xm = torch.mean(f_s, 0, keepdim=True) - f_s
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(f_t, 0, keepdim=True) - f_t
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4 * d * d)
        return loss
