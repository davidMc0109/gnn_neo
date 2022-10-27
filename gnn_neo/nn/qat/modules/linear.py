import torch
import torch.nn.functional as F

from gnn_neo.nn.qat.modules import _get_qconfig, _get_fake_quantize, _check_qconfig
from gnn_neo.quantization.qconfig import QConfig


class Linear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):

        # Extract kwarg 'qconfig' and pass the rest to super()
        self.qconfig = kwargs['qconfig']
        del kwargs['qconfig']

        super(Linear, self).__init__(*args, **kwargs)

        self.per_channel_qconfig = not isinstance(self.qconfig, QConfig)
        if self.per_channel_qconfig:
            # Separate out QConfig for each layer
            # `qconfig` is a dictionary with keys 0,1,2... and 'default'
            # The key refers to the output-channel number

            self.qconfig_list = [None]*self.out_features
            self.input_quantize = [None]*self.out_features
            self.weight_quantize = [None]*self.out_features
            self.bias_quantize = [None]*self.out_features
            self.output_quantize = [None]*self.out_features

            for i in range(self.out_features):

                # Get key. i if its a valid key, else 'default'
                # Check if keys are valid
                key = 'default'
                if i in self.qconfig:
                    key = i
                assert key in self.qconfig
                assert _check_qconfig(self.qconfig[key])

                self.qconfig_list[i] = self.qconfig[key]
                self.input_quantize[i], self.weight_quantize[i], \
                    self.bias_quantize[i], self.output_quantize[i] = \
                    _get_fake_quantize(self.qconfig_list[i])

        else:
            # All output channels have the same quantization properties
            assert isinstance(self.qconfig, QConfig)
            self.input_quantize, self.weight_quantize, self.bias_quantize, self.output_quantize = \
                _get_fake_quantize(self.qconfig)

    def forward(self, input):

        if self.per_channel_qconfig:
            # Compute outputs for each output channel
            # Then concatenate into one tensor before output
            out_list = [None]*self.out_features

            for i in range(self.out_features):
                inp_q = self.input_quantize[i](input)
                weight_q = self.weight_quantize[i](self.weight[i])

                if self.bias is not None:
                    bias_q = self.bias_quantize[i](self.bias[i])
                    out = F.linear(inp_q, weight_q, bias_q)
                else:
                    out = F.linear(inp_q, weight_q)

                out_list[i] = self.output_quantize[i](out)

            # Stack all output channels into 1 tensor
            out = torch.stack(out_list, dim=-1)
            return out

        else:
            if self.bias is not None:
                return self.output_quantize(
                    F.linear(self.input_quantize(input), self.weight_quantize(self.weight), self.bias_quantize(self.bias)))
            else:
                return self.output_quantize(F.linear(self.input_quantize(input), self.weight_quantize(self.weight)))


