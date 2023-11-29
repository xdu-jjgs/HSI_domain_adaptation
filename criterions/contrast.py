import torch
import torch.nn as nn


class SupInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupInfoNCELoss, self).__init__()
        assert temperature > 0.
        self.temperature = temperature

    def forward(self, features, labels):
        features = torch.squeeze(features)
        labels = labels.unsqueeze(1)

        self_contrast_mask = torch.eye(features.size(0)).to(features.device)
        self_contrast_mask = 1 - self_contrast_mask
        mask = torch.eq(labels, labels.T).float().to(features.device)
        mask = mask * self_contrast_mask

        similarity_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        assert self_contrast_mask.shape == similarity_matrix.shape == mask.shape
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        exp_logits = torch.exp(logits) * self_contrast_mask

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        # print(similarity_matrix, torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6).flatten())
        loss = -mean_log_prob_pos.mean()
        return loss
