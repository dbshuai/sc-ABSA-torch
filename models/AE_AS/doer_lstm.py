"""
使用Bi-LSTM代替ReGU的DOER，似乎收敛更好一些
"""
import sys

sys.path.append("./")
import torch
import torch.nn as nn
from allennlp.modules.conditional_random_field import ConditionalRandomField
from models.layers.CSU import Cross_Shared_Unit
from models.layers.dynamic_rnn import DynamicRNN


class DualCrossSharedLSTM(nn.Module):
    def __init__(self, general_embeddings, domain_embeddings, input_size, hidden_size, aspect_tag_classes,
                 polarity_tag_classes, k, dropout=0.5):
        super(DualCrossSharedLSTM, self).__init__()
        self.general_embedding = nn.Embedding(num_embeddings=general_embeddings.size(0),
                                              embedding_dim=general_embeddings.size(1),
                                              padding_idx=0).from_pretrained(general_embeddings)
        self.domain_embedding = nn.Embedding(num_embeddings=domain_embeddings.size(0),
                                             embedding_dim=domain_embeddings.size(1),
                                             padding_idx=0).from_pretrained(domain_embeddings)
        self.general_embedding.weight.requires_grad = False
        self.domain_embedding.weight.requires_grad = False

        self.dropout = dropout

        self.aspect_rnn1 =  DynamicRNN(input_size,hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.polarity_rnn1 = DynamicRNN(input_size,hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        self.csu = Cross_Shared_Unit(k, 2 * hidden_size)

        self.aspect_rnn2 =  DynamicRNN(hidden_size*2,hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.polarity_rnn2 = DynamicRNN(hidden_size*2,hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        self.aspect_hidden2tag = nn.Linear(2 * hidden_size, aspect_tag_classes)
        self.polarity_hidden2tag = nn.Linear(2 * hidden_size, polarity_tag_classes)

        self.aspect_crf = ConditionalRandomField(aspect_tag_classes)
        self.polarity_crf = ConditionalRandomField(polarity_tag_classes)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, features, aspect_tags, polarity_tags, mask,lengths, testing=False):
        general_features = self.general_embedding(features)
        domain_features = self.domain_embedding(features)
        features = torch.cat((general_features, domain_features), dim=2)
        max_len = torch.max(lengths)
        mask = mask[:,:max_len]
        features = features[:,:max_len,:]
        features = self.dropout_layer(features)
        aspect_hidden,(hn,cn) = self.aspect_rnn1(features,lengths)
        polarity_hidden,(hn,cn) = self.polarity_rnn1(features,lengths)

        #CSU
        aspect_hidden, polarity_hidden = self.csu(aspect_hidden, polarity_hidden, max_pooling=False)

        aspect_hidden_,(hn,cn) = self.aspect_rnn2(aspect_hidden,lengths)
        polarity_hidden_,(hn,cn) = self.polarity_rnn2(polarity_hidden,lengths)

        #res

        aspect_hidden = aspect_hidden_ + aspect_hidden
        polarity_hidden = polarity_hidden_ + polarity_hidden

        aspect_logit = self.aspect_hidden2tag(aspect_hidden)
        polarity_logit = self.polarity_hidden2tag(polarity_hidden)

        if testing == False:
            aspect_score = -self.aspect_crf(aspect_logit, aspect_tags, mask)
            polarity_score = -self.polarity_crf(polarity_logit, polarity_tags, mask)
            return aspect_score + polarity_score
        else:
            aspect_path = self.aspect_crf.viterbi_tags(aspect_logit, mask)
            polarity_path = self.polarity_crf.viterbi_tags(polarity_logit, mask)
            return aspect_path, polarity_path

