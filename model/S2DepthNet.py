import torch.nn as nn
import torch
import torch.nn.functional as F

from model.model import BaseERGB2Depth
from model.encoder_transformer import LongSpikeStreamEncoderConv
from model.submodules import ResidualBlock, ConvLayer, UpsampleConvLayer


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


def identity(x1, x2=None):
    return x1



class S2DepthTransformerUNetConv(BaseERGB2Depth):
    def __init__(self, config):
        super(S2DepthTransformerUNetConv, self).__init__(config)
        assert self.base_num_channels % 48 == 0

        self.depths=[int(i) for i in config["swin_depths"]]
        self.num_encoders = len(self.depths)
        self.num_heads=[int(i) for i in config["swin_num_heads"]]
        self.patch_size=[int(i) for i in config["swin_patch_size"]]
        self.out_indices=[int(i) for i in config["swin_out_indices"]]
        self.ape=config["ape"]
        try:
            self.num_v = config["new_v"]
        except KeyError:
            self.num_v = 0

        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders-1)
        self.activation = getattr(torch, 'sigmoid')
        # self.num_channel_spikes = config["num_channel_spikes"]
        self.num_output_channels = 1

        print('----- ', self.num_heads)
        self.encoder = LongSpikeStreamEncoderConv(
            patch_size=self.patch_size,
            in_chans=self.num_bins_rgb,
            embed_dim=self.base_num_channels,
            depths=self.depths,
            num_heads=self.num_heads,
            out_indices=self.out_indices,
            new_version=self.num_v,
        )

        self.UpsampleLayer = UpsampleConvLayer

        if self.skip_type == 'sum':
            self.apply_skip_connection = skip_sum
        elif self.skip_type == 'concat':
            self.apply_skip_connection = skip_concat
        elif self.skip_type == 'no_skip' or self.skip_type is None:
            self.apply_skip_connection = identity
        elif self.skip_type == 'attention':
            self.apply_skip_connection = self.skip_attention
        else:
            raise KeyError('Could not identify skip_type, please add "skip_type":'
                           ' "sum", "concat" or "no_skip" to config["model"]')

        self.build_resblocks()
        self.build_decoders()
        if self.skip_type == 'attention':
            self.build_attentions()
        self.build_prediction_layer()

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))
    

    def build_decoders(self):
        self.decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i) for i in range(self.num_encoders)]))
        # print(self.decoder_input_sizes)

        self.decoders = nn.ModuleList()
        first_decoder = True
        for input_size in self.decoder_input_sizes:
            if first_decoder:
                self.decoders.append(self.UpsampleLayer(input_size, input_size // 2,
                                                        kernel_size=5, padding=2, norm=self.norm))
                first_decoder = False
            else:
                self.decoders.append(self.UpsampleLayer(input_size if self.skip_type in ['sum', 'attention', 'no_skip']  else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))
        # print(self.decoders)

    def build_prediction_layer(self):
        '''
        self.pred = ConvLayer(self.base_num_channels // 2 if self.skip_type in ['sum', 'attention'] else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm)
                              '''
        self.pred = ConvLayer(self.base_num_channels // 2, self.num_output_channels, 1, activation=None, norm=self.norm)
    
    def build_attentions(self):
        self.channel_attentions = nn.ModuleList()
        self.attention_convs = nn.ModuleList()
        for input_size in self.decoder_input_sizes[:-1]:
            # print(input_size)
            self.channel_attentions.append(ChannelAttention(input_size))
            self.attention_convs.append(nn.Conv2d(input_size, input_size // 2, kernel_size=1))
        
    def skip_attention(self, x, y):
        combined_features = torch.cat((x, y), dim=1)
        num_channels = combined_features.shape[1]
        for i in range(self.num_encoders-1):
            if num_channels == self.decoder_input_sizes[i]:
                attention_weights = self.channel_attentions[i](combined_features)
                attention_applied = combined_features * attention_weights.expand_as(combined_features)
                return self.attention_convs[i](attention_applied)
                

    def forward_decoder(self, super_states):
        # last superstate is taken as input for decoder.
        if not bool(self.baseline) and self.state_combination == "convlstm":
            x = super_states[-1][0]
        else:
            x = super_states[-1]
        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            if i == 0:
                x = decoder(x)
            else:
                if not bool(self.baseline) and self.state_combination == "convlstm":
                    x = decoder(self.apply_skip_connection(x, super_states[self.num_encoders - i - 1][0]))
                else:
                    x = decoder(self.apply_skip_connection(x, super_states[self.num_encoders - i - 1]))

        img = self.activation(self.pred(x))

        return img

    def forward(self, item, prev_super_states, prev_states_lstm):
        #def forward(self, spike_tensor, prev_states=None):
        """
        :param spike_tensor: N x C x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """

        predictions_dict = {}

        spike_tensor = item["image"].to(self.gpu)
        encoded_xs = self.encoder(spike_tensor)
        # for x in encoded_xs:
            # print(x.shape)
        prediction = self.forward_decoder(encoded_xs)
        predictions_dict["image"] = prediction

        return predictions_dict, {'image': None}, prev_states_lstm

