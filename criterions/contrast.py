import torch
import torch.nn as nn


class SupInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, use_logit=False):
        super(SupInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.use_logit = use_logit

    def forward(self, features, labels):
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        self_contrast_mask = torch.eye(features.size(0)).to(features.device)
        self_contrast_mask = 1 - self_contrast_mask
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        assert self_contrast_mask.shape == similarity_matrix.shape == mask.shape
        mask = mask * self_contrast_mask

        if self.use_logit:
            logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
            logits = similarity_matrix - logits_max.detach()
        else:
            logits = similarity_matrix
        exp_logits = torch.exp(logits) * self_contrast_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        loss = -mean_log_prob_pos.mean()
        return loss
