import torch
import torch.nn as nn
from model.submodules import *

def concat(x1, x2):
    return torch.cat([x1, x2], dim=1)

def sum(x1, x2):
    return x1 + x2

base_num_channels = 96
num_encoders = 3
decoder_input_sizes = list(reversed([base_num_channels * pow(2, i) for i in range(num_encoders)]))
print(decoder_input_sizes)

decoders = nn.ModuleList()
first_decoder = True
skip_type = 'concat'

for input_size in decoder_input_sizes:
    if first_decoder:
        decoders.append(UpsampleConvLayer(input_size, input_size // 2, kernel_size=5, padding=2, norm="none"))
        first_decoder = False
    else:
        decoders.append(UpsampleConvLayer(input_size if skip_type in ['sum', 'attention']  else 2 * input_size,
                                            input_size // 2,
                                            kernel_size=5, padding=2, norm="none"))
print(decoders)


x = torch.rand((32, 384, 28, 28))
print(x.shape)

for i, decoder in enumerate(decoders):
    print(x.shape)
    print(decoder)
    if i == 0:
        x = decoder(x)
    else:
        x = decoder(sum(x, x) if skip_type=='sum' else concat(x, x))
        
print(x.shape)

pred = ConvLayer(base_num_channels // 2 if skip_type in ['sum', 'concat'] else 2 * base_num_channels, 1, 1, activation=None, norm="none")

print(pred)

x = pred(x)
print(x.shape)