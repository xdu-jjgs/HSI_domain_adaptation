import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        assert temperature > 0.
        self.temperature = temperature

    def forward(self, features, labels=None):
        """
        :param features: Tensor of shape [batch_size, feature_dim]
        :param labels: (optional) Ground truth labels. If not provided, the loss is calculated as InfoNCE without positive/negative pair supervision.
        """
        features = torch.squeeze(features)
        batch_size = features.size(0)

        # Compute similarity matrix
        similarity_matrix = torch.div(torch.matmul(features, features.T), self.temperature)

        # Remove diagonal entries (self-similarity)
        mask = torch.eye(batch_size).to(features.device)
        similarity_matrix = similarity_matrix * (1 - mask)

        # Compute logits
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute the softmax over all logits
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)

        if labels is not None:
            # If labels are provided, calculate the loss with supervision
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)
            # Calculate the mean log probability of positive samples
            mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
            loss = -mean_log_prob_pos.mean()
        else:
            # If no labels are provided, use InfoNCE loss without positive/negative supervision
            loss = -log_prob.mean()

        return loss
