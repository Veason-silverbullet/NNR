from DSSM_util import Config
import torch
import torch.nn as nn


class DSSM(nn.Module):
    def __init__(self, config: Config):
        super(DSSM, self).__init__()
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=config.DSSM_hidden_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.W3 = nn.Linear(in_features=config.DSSM_hidden_dim, out_features=config.DSSM_hidden_dim, bias=True)
        self.W4 = nn.Linear(in_features=config.DSSM_hidden_dim, out_features=config.DSSM_feature_dim, bias=True)

    def initialize(self):
        nn.init.uniform_(self.word_embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.W3.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.W4.weight, gain=nn.init.calculate_gain('tanh'))

    # Input
    # user_indices : [batch_size, sequence_length]
    # user_weights : [batch_size, sequence_length]
    # user_seq_len : [batch_size]
    # news_indices : [batch_size, news_num, sequence_length]
    # news_weights : [batch_size, news_num, sequence_length]
    # news_seq_len : [batch_size, news_num]
    # Output
    # logits       : [batch_size, news_num]
    def forward(self, user_indices, user_weights, user_seq_len, news_indices, news_weights, news_seq_len):
        user_embedding = self.dropout(self.word_embedding(user_indices) * user_weights.unsqueeze(dim=2)).sum(dim=1) # [batch_size, word_embeding_dim]
        news_embedding = self.dropout(self.word_embedding(news_indices) * news_weights.unsqueeze(dim=3)).sum(dim=2) # [batch_size, news_num, word_embeding_dim]
        user_l3 = self.dropout(torch.tanh(self.W3(user_embedding)))                                                 # [batch_size, hidden_dim]
        news_l3 = self.dropout(torch.tanh(self.W3(news_embedding)))                                                 # [batch_size, news_num, hidden_dim]
        user_y = self.dropout(torch.tanh(self.W4(user_l3)).unsqueeze(dim=1))                                        # [batch_size, 1, feature_dim]
        news_y = self.dropout(torch.tanh(self.W4(news_l3)))                                                         # [batch_size, news_num, feature_dim]
        norm = torch.norm(user_y, dim=2, keepdim=False) * torch.norm(news_y, dim=2, keepdim=False)                  # [batch_size, news_num]
        logits = (user_y * news_y).sum(dim=2, keepdim=False) / norm                                                 # [batch_size, news_num]
        return logits
