import torch
import torch.nn as nn


class OrthogonalDecomposedLoss(nn.Module):
    def __init__(self):
        super(OrthogonalDecomposedLoss, self).__init__()

    def forward(self, *vectors, **kwargs):
        loss = 0.0
        num_vectors = len(vectors)
        if num_vectors < 2:
            raise NotImplementedError("The length if vectors < 2!")
        for i in range(num_vectors):
            for j in range(i + 1, num_vectors):  # Avoid computing loss with itself and duplicate pairs
                a = vectors[i].squeeze()
                b = vectors[j].squeeze()
                a_norm = a / torch.norm(a, p=2, dim=1, keepdim=True)
                b_norm = b / torch.norm(b, p=2, dim=1, keepdim=True)
                M = a_norm * b_norm
                loss += torch.mean(torch.abs(torch.sum(M, dim=1)))
        return loss



