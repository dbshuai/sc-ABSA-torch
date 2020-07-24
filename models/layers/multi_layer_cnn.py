import torch
import torch.nn as nn


class MultiLayerCNN(nn.Module):
    def __init__(self, input_size, num_layer=4, dropout=0.5):
        super(MultiLayerCNN, self).__init__()
        self.num_layer = num_layer
        self.conv1 = nn.Conv1d(input_size, 128, 5, padding=2)  # (seq_len+2*padding-kernel_size)//strid + 1
        self.conv2 = nn.Conv1d(input_size, 128, 3, padding=1)  # padding是左右都padding

        self.dropout = nn.Dropout(dropout)
        if num_layer > 1:
            self.conv_seq = nn.ModuleList([Conv(dropout) for _ in range(num_layer-1)])

    def forward(self, x):
        """
        :param x: Tensor(batch_size,seq_len,input_size)
        :return:
        """
        x = self.dropout(x).transpose(1, 2)  # conv1d 在最后一维进行扫描
        x_conv = nn.functional.relu(torch.cat((self.conv1(x), self.conv2(x)), dim=1))
        if self.num_layer > 1:
            for conv in self.conv_seq:
                x_conv = conv(x_conv)
        x_conv = x_conv.transpose(1, 2)
        return x_conv


class Conv(nn.Module):
    def __init__(self, dropout=0.5):
        super(Conv, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(256, 256, 5, padding=2)

    def forward(self, x):
        x = self.dropout(x)
        x_conv = self.conv(x)
        x_conv = torch.relu(x_conv)
        return x_conv


if __name__ == '__main__':
    cnn = MultiLayerCNN(400,1,0.5)
    input = torch.randn(16,30,400)
    output = cnn(input)
    print(output.size())
