import torch.nn as nn
import torch.nn.init as init

def init_weights(model, nonlinearity='relu'):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
            init.constant_(m.bias, 0)