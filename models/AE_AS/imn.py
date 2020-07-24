import torch
import torch.nn as nn
import numpy as np
import torch.jit
from models.layers.multi_layer_cnn import MultiLayerCNN
from models.layers.attention import SelfAttention, DotSentimentAttention


class IMN(nn.Module):
    def __init__(self, gen_emb, domain_emb, ae_nums, as_nums, ds_nums, iters=4, dropout=0.5, use_opinion=True):
        """
        :param gen_emb: 通用词向量权重
        :param domain_emb: 领域词向量权重
        :param ae_nums: aspect和opinion word抽取的标签种类 int
        :param as_nums: aspect sentiment种类 int
        :param ds_nums: doc sentiment种类 int
        :param iters: message passing轮数 int
        :param dropout: float
        :param use_opinion: AE和AS之间是否建立联系 bool
        """
        super(IMN, self).__init__()
        self.iters = iters
        self.use_opinion = use_opinion
        self.dropout = nn.Dropout(dropout)
        # f_s
        self.general_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.general_embedding.weight = torch.nn.Parameter(gen_emb, requires_grad=False)
        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight = torch.nn.Parameter(domain_emb, requires_grad=False)
        self.encoder_shared = MultiLayerCNN(400, 2, dropout=dropout)

        # f_ae
        self.encoder_aspect = MultiLayerCNN(256, 2, dropout=dropout)
        self.decoder_aspect = Decoder(912, ae_nums)

        # f_as
        self.att_sentiment = SelfAttention(256, use_opinion=self.use_opinion)
        self.encoder_sentiment = MultiLayerCNN(256, 0, dropout=dropout)
        self.decoder_sentiment = Decoder(512, as_nums)

        """
        原代码中是kernel为5的CNN，此处为kernel为5和3的CNN拼接
        该部分为使用document数据完成领域预测和情感预测，目的是补偿AEAS任务标注数据不足
        """
        # f_ds
        self.encoder_doc_sentiment = MultiLayerCNN(256, 0, dropout=dropout)
        self.att_doc_sentiment = DotSentimentAttention(256)
        self.decoder_doc_sentiment = nn.Linear(256, ds_nums)
        # f_dd
        self.encoder_doc_domain = MultiLayerCNN(256, 0, dropout=dropout)
        self.att_doc_domain = DotSentimentAttention(256)
        self.decoder_doc_domain = nn.Linear(256, 2)
        # update
        self.update = nn.Linear(256 + as_nums + ae_nums + ds_nums + 2, 256)

    def emb(self, features):
        general_features = self.general_embedding(features)
        domain_features = self.domain_embedding(features)
        features = torch.cat((general_features, domain_features), dim=2)
        return features

    def forward(self, feature, op_label_feature=None, p_gold_op=None, mask=None, doc_training=False):
        """
        :param feature: batch,sent_len doc部分与AE、AS部分是分开训练的，feature可以是doc输入，也可以是AE、AS任务输入，训练时需要将任务无关参数冻结
        :param op_label_feature: batch,sent_len,3
        :param p_gold_op: batch,sent_len
        :param mask: batch,sent_len
        :return:
        """
        # sentiment shared layers
        feature_emb = self.emb(feature)  # batch,sent_len,domain+general=400
        feature_shared = self.encoder_shared(feature_emb)  # batch,sent_len,256
        feature_emb = torch.cat((feature_emb, feature_shared), dim=-1)  # batch,sent_len,656
        init_shared_features = feature_shared  # batch,sent_len,256

        aspect_probs = None
        sentiment_probs = None
        doc_senti_probs = None
        doc_domain_probs = None
        # task specfic layers
        for i in range(self.iters):
            aspect_output = feature_shared
            sentiment_output = feature_shared
            ### DS ###
            doc_senti_output = self.encoder_doc_sentiment(feature_shared)
            senti_att_weights_softmax, senti_att_weights_sigmoid = self.att_doc_sentiment(doc_senti_output)
            # batch,doc_len
            senti_weights = senti_att_weights_sigmoid.unsqueeze(-1)  # 1
            doc_senti_output = torch.sum(doc_senti_output * (senti_att_weights_softmax.unsqueeze(2)), dim=1)
            doc_senti_output = self.dropout(doc_senti_output)
            doc_senti_probs = self.decoder_doc_sentiment(doc_senti_output)
            doc_senti_probs_cat = doc_senti_probs.unsqueeze(1).repeat(1, feature_shared.size(1), 1)  # nums_DS

            ### DD ###
            doc_domain_output = self.encoder_doc_domain(feature_shared)
            doc_domain_weights_softmax, doc_domain_weights_sigmoid = self.att_doc_domain(doc_domain_output)
            domain_weights = doc_domain_weights_sigmoid.unsqueeze(-1)
            doc_domain_output = torch.sum(doc_domain_output * (senti_att_weights_softmax.unsqueeze(2)), dim=1)
            doc_domain_output = self.dropout(doc_domain_output)
            doc_domain_probs = self.decoder_doc_domain(doc_domain_output)
            # 256+5+5+3+1+1
            ### AE ###
            aspect_output = self.encoder_aspect(
                aspect_output)  # batch,sent_len,256 CNN的kernel\padding\strid刚好保证了sent_len不变
            aspect_output = torch.cat((feature_emb, aspect_output), dim=-1)  # batch,sent_len,656+256
            aspect_output = self.dropout(aspect_output)
            aspect_probs = self.decoder_aspect(aspect_output)
            ### AS ###
            sentiment_output = self.encoder_sentiment(sentiment_output)  # batch,sent_len,256
            sentiment_output = self.att_sentiment(sentiment_output, op_label_feature, aspect_probs,
                                                  p_gold_op)  # batch,sent_len,256
            sentiment_output = torch.cat((init_shared_features, sentiment_output), dim=-1)  # batch,sent_len,256+256
            sentiment_output = self.dropout(sentiment_output)
            sentiment_probs = self.decoder_sentiment(sentiment_output)
            feature_shared = torch.cat((feature_shared, aspect_probs, sentiment_probs, doc_senti_probs_cat,
                                        senti_weights, domain_weights), dim=-1)
            feature_shared = self.update(feature_shared)
        return aspect_probs, sentiment_probs, doc_senti_probs, doc_domain_probs

class Decoder(nn.Module):
    def __init__(self,input_size,tag_nums):
        super(Decoder,self).__init__()
        self.dense = nn.Linear(input_size,tag_nums)

    def forward(self,x):
        x_logit = self.dense(x)
        y = torch.log_softmax(x_logit,dim=-1)
        return y

if __name__ == '__main__':
    gen_emb = torch.randn(400, 300)
    domain_emb = torch.randn(400, 100)
    ae_nums, as_nums, ds_nums, dd_nums = 5, 5, 3, 1
    imn = IMN(gen_emb, domain_emb, ae_nums, as_nums, ds_nums, iters=2)
    feature = torch.LongTensor([2, 3, 4, 1, 1, 0]).unsqueeze(0).repeat(3, 1)
    doc = feature
    op_label_feature = torch.FloatTensor([[[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]]).repeat(3,
                                                                                                                      1,
                                                                                                                      1)
    mask = None
    p_gold_op = torch.FloatTensor([[1, 0, 1, 1, 0, 1]]).repeat(3, 1)
    aspect_probs, sentiment_probs, doc_senti_probs, doc_domain_probs = imn(feature, op_label_feature, p_gold_op, mask,
                                                                           doc_training=True)
