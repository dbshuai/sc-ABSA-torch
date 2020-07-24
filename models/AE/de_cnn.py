import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, num_classes=3, dropout=0.5, crf=False):
        super(Model, self).__init__()
        self.gen_embedding = nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight = nn.Parameter(gen_emb, requires_grad=False)
        self.domain_embedding = nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight = nn.Parameter(domain_emb, requires_grad=False)

        self.conv1 = nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 5, padding=2)
        self.conv2 = nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(256, 256, 5, padding=2)
        self.conv4 = nn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = nn.Conv1d(256, 256, 5, padding=2)
        self.linear_ae = nn.Linear(256, num_classes)
        self.crf_flag = crf
        if self.crf_flag:
            from allennlp.modules import ConditionalRandomField
            self.crf = ConditionalRandomField(num_classes)

    def forward(self, x,y, x_len, x_mask, x_tag=None, testing=False):
        x_emb = torch.cat((self.gen_embedding(x), self.domain_embedding(y)), dim=2)
        x_emb = self.dropout(x_emb).transpose(1, 2)
        x_conv = nn.functional.relu(torch.cat((self.conv1(x_emb), self.conv2(x_emb)), dim=1))
        x_conv = self.dropout(x_conv)
        x_conv = nn.functional.relu(self.conv3(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = nn.functional.relu(self.conv4(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = nn.functional.relu(self.conv5(x_conv))
        x_conv = x_conv.transpose(1, 2)
        x_logit = self.linear_ae(x_conv)
        if testing:
            if self.crf_flag:
                score = self.crf.viterbi_tags(x_logit, x_mask)
            else:
                x_logit = x_logit.transpose(2, 0)
                score = nn.functional.log_softmax(x_logit).transpose(2, 0)
        else:
            if self.crf_flag:
                score = -self.crf(x_logit, x_tag, x_mask)
            else:
                x_len_sorted, sort = torch.sort(x_len, dim=0, descending=True)
                x_logit_sorted = torch.index_select(x_logit, dim=0, index=sort)
                x_tag_sorted = torch.index_select(x_tag,dim=0,index=sort)
                x_logit = nn.utils.rnn.pack_padded_sequence(x_logit_sorted, x_len_sorted, batch_first=True)
                x_tag = nn.utils.rnn.pack_padded_sequence(x_tag_sorted, x_len_sorted, batch_first=True)
                score = nn.functional.nll_loss(nn.functional.log_softmax(x_logit.data), x_tag.data)
        return score