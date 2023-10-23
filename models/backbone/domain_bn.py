import torch.nn as nn
from models.utils.init import initialize_weights


def switch_domain(model: nn.Module, domain_id: int):
    for m in model.modules():
        if isinstance(m, DomainBatchNorm2d):
            m.set_domain(domain_id)


class DomainBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, num_domains: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_features > 1
        self.num_domains = num_domains
        self.bn_domains = nn.ModuleList([nn.BatchNorm2d(num_features) for _ in range(num_domains)])
        self.current_domain = 0
        initialize_weights(self.bn_domains)

    def forward(self, x):
        out = self.bn_domains[self.current_domain](x)
        return out

    def set_domain(self, domain_id):
        if self.current_domain != domain_id:
            print("switch domain {}".format(domain_id))
        self.current_domain = domain_id

