import torch
import torch.nn as nn
import torch.nn.init as init

from modules.blocks import MiniDownBlock, MiniUpBlock
from modules.resnets import MiniResNet

def init_weights(model, nonlinearity='relu'):
    for m in model.modules():
        # Ignore the model itself
        if type(m) == type(model):
            pass

        # Recurse into children
        elif isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
            init_weights(m, nonlinearity)

        # Initialize the named modules (no weights at top level)
        elif isinstance(m, nn.MultiheadAttention):
            init_weights(m, nonlinearity)
        elif isinstance(m, MiniDownBlock) or isinstance(m, MiniUpBlock):
            init_weights(m, nonlinearity)
        elif isinstance(m, MiniResNet):
            init_weights(m, nonlinearity)

        # Initialize common components
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
            init.constant_(m.bias, 0)

        # Ignore modules w/o learnable parameters
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

        # Blindly recurse into unknown modules
        elif not hasattr(m, 'weight'):
            init_weights(m, nonlinearity)

        else:
            raise ValueError(f'Unknown module type: {type(m)}')


def set_dropout_rate(model, new_dropout_rate):
    for layer in model.modules():
        if type(model) == type(model):
            pass
        elif isinstance(layer, nn.Dropout) or isinstance(layer, nn.Dropout2d) or isinstance(layer, nn.Dropout3d):
            layer.p = new_dropout_rate
        # Blindly recurse
        elif hasattr(layer, 'modules'):
            set_dropout_rate(layer, new_dropout_rate)


def reset_optimizer_state(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = torch.zeros_like(v)