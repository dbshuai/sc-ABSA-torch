"""
根据ABSA-pytorch中同名文件修改而成，方便调用LSTM、GRU、普通RNN三种RNN模型。主要是将pad、pack进行了包装
"""

import torch
import torch.nn as nn


class DynamicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """
        sen_outs,(hn,cn) = self.sen_rnn(sen_batch)
        # sen_outs = utils.rnn.pad_packed_sequence(sen_outs,batch_first=True) #batch_size,max_len,hid_dim*2
        # sen_outs = torch.index_select(sen_outs,dim=0,index=unsort)
        # sen_rnn = sen_outs.contiguous().view(batch_size, -1, 2 * self.hidden_dim)  # (batch, sen_len, 2*hid)

        ''' Fetch the truly last hidden layer of both sides
        """
        """sort"""
        x_len_sorted, sort = torch.sort(x_len, dim=0, descending=True)
        _, unsort = torch.sort(sort, dim=0)
        x = torch.index_select(x, dim=0, index=sort)
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len_sorted, batch_first=self.batch_first)

        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p)
        else:
            out_pack, ht = self.RNN(x_emb_p)
            ct = None
        """unsort: h"""
        ht = torch.index_select(ht, dim=1, index=unsort)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = torch.index_select(out, dim=0, index=unsort)
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.index_select(ct, dim=1, index=unsort)
            return out, (ht, ct)


if __name__ == '__main__':
    rnn = DynamicRNN(5, 10)
    input = torch.randn(6, 8, 5)
    x_len = torch.Tensor([5, 3, 6, 7, 5, 3])
    output = rnn(input, x_len)
