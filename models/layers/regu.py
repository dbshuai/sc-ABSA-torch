"""
可以快乐地使用jit进行加速的ReGU，在底部有用例
ReGU为DOER中提出的RNN，优势在于自带多层残差，在stacked-rnn做序列标注中占优势
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import List


def ReGU(input_size, hidden_size, num_layers, dropout=False, bidirectional=False):
    if bidirectional:
        print(1)
        stack_type = StackedReGU2
        layer_type = BidirReGULayer
        dirs = 2
    else:
        stack_type = StackedReGU
        layer_type = ReGULayer
        dirs = 1

    return stack_type(num_layers, layer_type,
                      first_layer_args=[ReGUCell, input_size, hidden_size],
                      other_layer_args=[ReGUCell, hidden_size * dirs,
                                        hidden_size])


class ReGUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ReGUCell, self).__init__()
        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_if = nn.Linear(input_size, hidden_size)
        self.W_io = nn.Linear(input_size, hidden_size)
        self.W_ix = nn.Linear(input_size, hidden_size)
        self.U_io = nn.Linear(hidden_size, hidden_size)
        self.U_if = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c):
        f_t = torch.sigmoid(self.W_if(x) + self.U_if(c))
        o_t = torch.sigmoid(self.W_io(x) + self.U_io(c))
        c_t = (1 - f_t) * c + f_t * torch.tanh(self.W_ii(x))
        if x.size() == c.size():
            h_t = (1 - o_t) * c_t + o_t * x
        else:
            h_t = (1 - o_t) * c_t + o_t * torch.tanh(self.W_ix(x))
        return h_t, c_t


class ReGULayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(ReGULayer, self).__init__()
        self.cell_args = cell_args
        self.cell = cell(*cell_args)

    def forward(self, inputs, state):
        outputs = torch.empty(0, inputs.size(1), self.cell_args[1]).to(inputs.device)
        for i in range(inputs.size(0)):
            out, state = self.cell(inputs[i], state)
            out = out.unsqueeze(0)
            outputs = torch.cat((outputs, out), dim=0)
        return outputs, state


class ReverseReGULayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(ReverseReGULayer, self).__init__()
        self.cell_args = cell_args
        self.cell = cell(*cell_args)

    def forward(self, inputs, state):
        inputs = inputs.flip(dims=[0, ])
        outputs = torch.empty(0, inputs.size(1), self.cell_args[1]).to(inputs.device)
        for i in range(inputs.size(0)):
            out, state = self.cell(inputs[i], state)
            out = out.unsqueeze(0)
            outputs = torch.cat((outputs, out), dim=0)
        outputs = outputs.flip(dims=[0, ])
        return outputs, state


class BidirReGULayer(nn.Module):

    def __init__(self, cell, *cell_args):
        super(BidirReGULayer, self).__init__()
        self.directions = nn.ModuleList([
            ReGULayer(cell, *cell_args),
            ReverseReGULayer(cell, *cell_args),
        ])

    def forward(self, input, states):
        outputs = []
        output_states = []
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states


def init_stacked_regu(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


class StackedReGU(nn.Module):

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedReGU, self).__init__()
        self.layers = init_stacked_regu(num_layers, layer, first_layer_args,
                                        other_layer_args)

    def forward(self, input, states):
        input = input.permute(1, 0, 2)
        output_states = torch.jit.annotate(List[List[Tensor,]], [])
        output = input
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            state = state.squeeze(0)
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        output = output.permute(1, 0, 2)
        return output, output_states


class StackedReGU2(nn.Module):

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedReGU2, self).__init__()
        self.layers = init_stacked_regu(num_layers, layer, first_layer_args,
                                        other_layer_args)

    def forward(self, input, states):
        input = input.permute(1, 0, 2)
        output_states = torch.jit.annotate(List[List[Tensor,]], [])
        output = input
        i = 0
        for rnn_layer in self.layers:
            state = states[i]  # layer x  dir x batch x hidden
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        output = output.permute(1, 0, 2)
        return output, output_states


if __name__ == '__main__':
    inp = torch.randn(16, 83, 400)
    states = torch.randn(2, 2, 16, 300)  # layer,dir,batch,hidden
    rnn1 = torch.jit.script(ReGU(400, 300, 2, bidirectional=True))  # 使用jit加速计算
    rnn2 = ReGU(400, 300, 2, bidirectional=True)
    import time

    t = time.time()
    out, out_state = rnn2(inp, states)
    print(time.time() - t)
    t = time.time()
    out, out_state = rnn1(inp, states)
    print(time.time() - t)
    print(len(out_state[0][0][0]))
