import torch
import torch.nn as nn


def save_grad(name, layer_gradients):
    def hook(module, grad_input, grad_output):
        if grad_output is not None and grad_output[0] is not None:
            layer_gradients[name] = grad_output[0].detach()

    return hook


def register_layer_hook(model: nn.Module):
    model.layer_gradients = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Module) and not isinstance(layer, nn.Sequential):
            layer.register_full_backward_hook(save_grad(name, model.layer_gradients))


'''
hook与监听者模式的区别：hook一般在预设的固定点触发（比如vue的生命周期）、而监听这模式为动态的、事件驱动的触发
'''
