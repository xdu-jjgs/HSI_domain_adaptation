import torch
import torch.nn as nn

from fvcore.nn import FlopCountAnalysis


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters()) / 1_000_000  # 将参数量转换为百万（M）


def count_flops(model: nn.Module, inputs: torch.Tensor, *args):
    bs = inputs.size()[0]
    flop_count = FlopCountAnalysis(model, (inputs, *args))
    return flop_count.total() / bs / 1e6
