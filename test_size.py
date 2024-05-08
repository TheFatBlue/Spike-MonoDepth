import sys
import torch
import torch.nn as nn
from model.submodules import *
from model.encoder_transformer import LongSpikeStreamEncoderConv

def concat(x1, x2):
    return torch.cat([x1, x2], dim=1)

def sum(x1, x2):
    return x1 + x2

base_num_channels = 96
num_encoders = 4

encoder = LongSpikeStreamEncoderConv(
    patch_size=(32,2,2),
    in_chans=128,
    embed_dim=96,
    depths=[2,2,6,6],
    num_heads=[3,6,12,24],
    out_indices=(0,1,2,3),
    new_version=4,
)

max_num_channels = base_num_channels * pow(2, num_encoders-1)

resblocks = nn.ModuleList()
for i in range(2):
    resblocks.append(ResidualBlock(max_num_channels, max_num_channels, norm="none"))

decoder_input_sizes = list(reversed([base_num_channels * pow(2, i) for i in range(num_encoders)]))
print(decoder_input_sizes)

decoders = nn.ModuleList()
first_decoder = True
skip_type = 'sum'

for input_size in decoder_input_sizes:
    if first_decoder:
        decoders.append(UpsampleConvLayer(input_size, input_size // 2, kernel_size=5, padding=2, norm="none"))
        first_decoder = False
    else:
        decoders.append(UpsampleConvLayer(input_size if skip_type in ['sum', 'attention']  else 2 * input_size,
                                            input_size // 2,
                                            kernel_size=5, padding=2, norm="none"))
print(decoders)


x = torch.rand((1, 128, 224, 224))

encoded_xs = encoder(x)

x = encoded_xs[-1]

for resblock in resblocks:
    x = resblock(x)

for i, decoder in enumerate(decoders):
    if i == 0:
        x = decoder(x)
    else:
        x = decoder(sum(x, x) if skip_type=='sum' else concat(x, x))
        


pred = ConvLayer(base_num_channels // 2 if skip_type in ['sum', 'concat'] else 2 * base_num_channels, 1, 1, activation=None, norm="none")

print(pred)

x = pred(x)
print(x.shape)