import math
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from layers import Conv1D, Attention, ScaledDotProduct_CandidateAttention, GCN
from torch_scatter import scatter_sum, scatter_softmax # need to be installed by following `https://pytorch-scatter.readthedocs.io/en/latest`
from newsEncoders import NewsEncoder
from userEncoders import UserEncoder


class CNE_Title(NewsEncoder):
    def __init__(self, config: Config):
        super(CNE_Title, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.word_embedding_dim = config.word_embedding_dim
        self.hidden_dim = config.hidden_dim
        self.news_embedding_dim = config.hidden_dim * 2 + config.category_embedding_dim + config.subCategory_embedding_dim
        # LSTM encoder
        self.title_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        # self-attention
        self.title_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)

    def initialize(self):
        super().initialize()
        for parameter in self.title_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        self.title_self_attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        title_mask = title_mask.view([batch_news_num, self.max_title_length])                                                         # [batch_size * news_num, max_title_length]
        title_mask[:, 0] = 1 # To avoid empty input of LSTM
        title_length = title_mask.sum(dim=1, keepdim=False).long()                                                                    # [batch_size * news_num]
        sorted_title_length, sorted_title_indices = torch.sort(title_length, descending=True)                                         # [batch_size * news_num]
        _, desorted_title_indices = torch.sort(sorted_title_indices, descending=False)                                                # [batch_size * news_num]
        # 1. word embedding
        title = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])  # [batch_size * news_num, max_title_length, word_embedding_dim]
        sorted_title = pack_padded_sequence(title.index_select(0, sorted_title_indices), sorted_title_length.cpu(), batch_first=True) # [batch_size * news_num, max_title_length, word_embedding_dim]
        # 2. LSTM encoding
        sorted_title_h, (sorted_title_h_n, sorted_title_c_n) = self.title_lstm(sorted_title)
        sorted_title_h, _ = pad_packed_sequence(sorted_title_h, batch_first=True, total_length=self.max_title_length)                 # [batch_size * news_num, max_title_length, hidden_dim * 2]
        title_h = sorted_title_h.index_select(0, desorted_title_indices)                                                              # [batch_size * news_num, max_title_length, hidden_dim * 2]
        # 3. self-attention
        title_self = self.title_self_attention(title_h, title_mask).view([batch_size, news_num, self.hidden_dim * 2])                 # [batch_size * news_num, hidden_dim * 2]
        # 4. feature fusion
        news_representation = self.feature_fusion(title_self, category, subCategory)                                                  # [batch_size, news_num, news_embedding_dim]
        return news_representation


class CNE_Content(NewsEncoder):
    def __init__(self, config: Config):
        super(CNE_Content, self).__init__(config)
        self.max_content_length = config.max_abstract_length
        self.word_embedding_dim = config.word_embedding_dim
        self.hidden_dim = config.hidden_dim
        self.news_embedding_dim = config.hidden_dim * 2 + config.category_embedding_dim + config.subCategory_embedding_dim
        # LSTM encoder
        self.content_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        # self-attention
        self.content_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)

    def initialize(self):
        super().initialize()
        for parameter in self.content_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        self.content_self_attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        content_mask = content_mask.view([batch_news_num, self.max_content_length])                                                           # [batch_size * news_num, max_content_length]
        content_mask[:, 0] = 1 # To avoid empty input of LSTM
        content_length = content_mask.sum(dim=1, keepdim=False).long()                                                                        # [batch_size * news_num]
        sorted_content_length, sorted_content_indices = torch.sort(content_length, descending=True)                                           # [batch_size * news_num]
        _, desorted_content_indices = torch.sort(sorted_content_indices, descending=False)                                                    # [batch_size * news_num]
        # 1. word embedding
        content = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim])    # [batch_size * news_num, max_content_length, word_embedding_dim]
        sorted_content = pack_padded_sequence(content.index_select(0, sorted_content_indices), sorted_content_length.cpu(), batch_first=True) # [batch_size * news_num, max_content_length, word_embedding_dim]
        # 2. LSTM encoding
        sorted_content_h, (sorted_content_h_n, sorted_content_c_n) = self.content_lstm(sorted_content)
        sorted_content_h, _ = pad_packed_sequence(sorted_content_h, batch_first=True, total_length=self.max_content_length)                   # [batch_size * news_num, max_content_length, hidden_dim * 2]
        content_h = sorted_content_h.index_select(0, desorted_content_indices)                                                                # [batch_size * news_num, max_content_length, hidden_dim * 2]
        # 3. self-attention
        content_self = self.content_self_attention(content_h, content_mask).view([batch_size, news_num, self.hidden_dim * 2])                 # [batch_size * news_num, hidden_dim * 2]
        # 4. feature fusion
        news_representation = self.feature_fusion(content_self, category, subCategory)                                                        # [batch_size, news_num, news_embedding_dim]
        return news_representation


class NAML_Title(NewsEncoder):
    def __init__(self, config: Config):
        super(NAML_Title, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.news_embedding_dim = config.cnn_kernel_num
        self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(config.category_embedding_dim, config.cnn_kernel_num, bias=True)
        self.subCategory_affine = nn.Linear(config.subCategory_embedding_dim, config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)

    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.subCategory_affine.weight)
        nn.init.zeros_(self.subCategory_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word embedding
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim]) # [batch_size * news_num, word_embedding_dim, max_title_length]
        # 2. CNN encoding
        title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))                                            # [batch_size * news_num, max_title_length, cnn_kernel_num]
        # 3. attention layer
        title_representation = self.title_attention(title_c).view([batch_size, news_num, self.cnn_kernel_num])                         # [batch_size, news_num, cnn_kernel_num]
        # 4. category and subCategory encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)                        # [batch_size, news_num, cnn_kernel_num]
        subCategory_representation = F.relu(self.subCategory_affine(self.subCategory_embedding(subCategory)), inplace=True)            # [batch_size, news_num, cnn_kernel_num]
        # 5. multi-view attention
        feature = torch.stack([title_representation, category_representation, subCategory_representation], dim=2)                      # [batch_size, news_num, 3, cnn_kernel_num]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)                                                      # [batch_size, news_num, 3, 1]
        news_representation = (feature * alpha).sum(dim=2, keepdim=False)                                                              # [batch_size, news_num, cnn_kernel_num]
        return news_representation


class NAML_Content(NewsEncoder):
    def __init__(self, config: Config):
        super(NAML_Content, self).__init__(config)
        self.max_content_length = config.max_abstract_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.news_embedding_dim = config.cnn_kernel_num
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(config.category_embedding_dim, config.cnn_kernel_num, bias=True)
        self.subCategory_affine = nn.Linear(config.subCategory_embedding_dim, config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)

    def initialize(self):
        super().initialize()
        self.content_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.subCategory_affine.weight)
        nn.init.zeros_(self.subCategory_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word embedding
        content_w = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim]) # [batch_size * news_num, max_content_length, word_embedding_dim]
        # 2. CNN encoding
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))                                            # [batch_size * news_num, max_content_length, cnn_kernel_num]
        # 3. attention layer
        content_representation = self.content_attention(content_c).view([batch_size, news_num, self.cnn_kernel_num])                         # [batch_size, news_num, cnn_kernel_num]
        # 4. category and subCategory encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)                              # [batch_size, news_num, cnn_kernel_num]
        subCategory_representation = F.relu(self.subCategory_affine(self.subCategory_embedding(subCategory)), inplace=True)                  # [batch_size, news_num, cnn_kernel_num]
        # 5. multi-view attention
        feature = torch.stack([content_representation, category_representation, subCategory_representation], dim=2)                          # [batch_size, news_num, 3, cnn_kernel_num]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)                                                            # [batch_size, news_num, 3, 1]
        news_representation = (feature * alpha).sum(dim=2, keepdim=False)                                                                    # [batch_size, news_num, cnn_kernel_num]
        return news_representation


class CNE_wo_CS(NewsEncoder):
    def __init__(self, config: Config):
        super(CNE_wo_CS, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.word_embedding_dim = config.word_embedding_dim
        self.hidden_dim = config.hidden_dim
        self.news_embedding_dim = config.hidden_dim * 4 + config.category_embedding_dim + config.subCategory_embedding_dim
        # LSTM encoder
        self.title_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.content_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        # self-attention
        self.title_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)
        self.content_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)
        # cross-attention
        self.title_cross_attention = ScaledDotProduct_CandidateAttention(self.hidden_dim * 2, self.hidden_dim * 2, config.attention_dim)
        self.content_cross_attention = ScaledDotProduct_CandidateAttention(self.hidden_dim * 2, self.hidden_dim * 2, config.attention_dim)

    def initialize(self):
        super().initialize()
        for parameter in self.title_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        for parameter in self.content_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        self.title_self_attention.initialize()
        self.content_self_attention.initialize()
        self.title_cross_attention.initialize()
        self.content_cross_attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        title_mask = title_mask.view([batch_news_num, self.max_title_length])                                                                              # [batch_size * news_num, max_title_length]
        content_mask = content_mask.view([batch_news_num, self.max_content_length])                                                                        # [batch_size * news_num, max_content_length]
        title_mask[:, 0] = 1   # To avoid empty input of LSTM
        content_mask[:, 0] = 1 # To avoid empty input of LSTM
        title_length = title_mask.sum(dim=1, keepdim=False).long()                                                                                         # [batch_size * news_num]
        content_length = content_mask.sum(dim=1, keepdim=False).long()                                                                                     # [batch_size * news_num]
        sorted_title_length, sorted_title_indices = torch.sort(title_length, descending=True)                                                              # [batch_size * news_num]
        _, desorted_title_indices = torch.sort(sorted_title_indices, descending=False)                                                                     # [batch_size * news_num]
        sorted_content_length, sorted_content_indices = torch.sort(content_length, descending=True)                                                        # [batch_size * news_num]
        _, desorted_content_indices = torch.sort(sorted_content_indices, descending=False)                                                                 # [batch_size * news_num]
        # 1. word embedding
        title = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])                       # [batch_size * news_num, max_title_length, word_embedding_dim]
        content = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim])                 # [batch_size * news_num, max_content_length, word_embedding_dim]
        sorted_title = pack_padded_sequence(title.index_select(0, sorted_title_indices), sorted_title_length.cpu(), batch_first=True)                      # [batch_size * news_num, max_title_length, word_embedding_dim]
        sorted_content = pack_padded_sequence(content.index_select(0, sorted_content_indices), sorted_content_length.cpu(), batch_first=True)              # [batch_size * news_num, max_content_length, word_embedding_dim]
        # 2. LSTM encoding
        sorted_title_h, (sorted_title_h_n, sorted_title_c_n) = self.title_lstm(sorted_title)
        sorted_content_h, (sorted_content_h_n, sorted_content_c_n) = self.content_lstm(sorted_content)
        sorted_title_h, _ = pad_packed_sequence(sorted_title_h, batch_first=True, total_length=self.max_title_length)                                      # [batch_size * news_num, max_title_length, hidden_dim * 2]
        sorted_content_h, _ = pad_packed_sequence(sorted_content_h, batch_first=True, total_length=self.max_content_length)                                # [batch_size * news_num, max_content_length, hidden_dim * 2]
        title_h = sorted_title_h.index_select(0, desorted_title_indices)                                                                                   # [batch_size * news_num, max_title_length, hidden_dim * 2]
        content_h = sorted_content_h.index_select(0, desorted_content_indices)                                                                             # [batch_size * news_num, max_content_length, hidden_dim * 2]
        # 3. self-attention
        title_self = self.title_self_attention(title_h, title_mask)                                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_self = self.content_self_attention(content_h, content_mask)                                                                                # [batch_size * news_num, hidden_dim * 2]
        # 4. cross-attention
        title_cross = self.title_cross_attention(title_h, content_self, title_mask)                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_cross = self.content_cross_attention(content_h, title_self, content_mask)                                                                  # [batch_size * news_num, hidden_dim * 2]
        news_representation = torch.cat([title_self + title_cross, content_self + content_cross], dim=1).view([batch_size, news_num, self.hidden_dim * 4]) # [batch_size * news_num, hidden_dim * 4]
        # 5. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                                              # [batch_size, news_num, news_embedding_dim]
        return news_representation


class CNE_wo_CA(NewsEncoder):
    def __init__(self, config: Config):
        super(CNE_wo_CA, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.word_embedding_dim = config.word_embedding_dim
        self.hidden_dim = config.hidden_dim
        self.news_embedding_dim = config.hidden_dim * 4 + config.category_embedding_dim + config.subCategory_embedding_dim
        # selective LSTM encoder
        self.title_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.content_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.title_H = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.title_M = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=True)
        self.content_H = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.content_M = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=True)
        # self-attention
        self.title_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)
        self.content_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)

    def initialize(self):
        super().initialize()
        for parameter in self.title_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        for parameter in self.content_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        nn.init.xavier_uniform_(self.title_H.weight)
        nn.init.xavier_uniform_(self.title_M.weight)
        nn.init.zeros_(self.title_M.bias)
        nn.init.xavier_uniform_(self.content_H.weight)
        nn.init.xavier_uniform_(self.content_M.weight)
        nn.init.zeros_(self.content_M.bias)
        self.title_self_attention.initialize()
        self.content_self_attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        title_mask = title_mask.view([batch_news_num, self.max_title_length])                                                                 # [batch_size * news_num, max_title_length]
        content_mask = content_mask.view([batch_news_num, self.max_content_length])                                                           # [batch_size * news_num, max_content_length]
        title_mask[:, 0] = 1   # To avoid empty input of LSTM
        content_mask[:, 0] = 1 # To avoid empty input of LSTM
        title_length = title_mask.sum(dim=1, keepdim=False).long()                                                                            # [batch_size * news_num]
        content_length = content_mask.sum(dim=1, keepdim=False).long()                                                                        # [batch_size * news_num]
        sorted_title_length, sorted_title_indices = torch.sort(title_length, descending=True)                                                 # [batch_size * news_num]
        _, desorted_title_indices = torch.sort(sorted_title_indices, descending=False)                                                        # [batch_size * news_num]
        sorted_content_length, sorted_content_indices = torch.sort(content_length, descending=True)                                           # [batch_size * news_num]
        _, desorted_content_indices = torch.sort(sorted_content_indices, descending=False)                                                    # [batch_size * news_num]
        # 1. word embedding
        title = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])          # [batch_size * news_num, max_title_length, word_embedding_dim]
        content = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim])    # [batch_size * news_num, max_content_length, word_embedding_dim]
        sorted_title = pack_padded_sequence(title.index_select(0, sorted_title_indices), sorted_title_length.cpu(), batch_first=True)         # [batch_size * news_num, max_title_length, word_embedding_dim]
        sorted_content = pack_padded_sequence(content.index_select(0, sorted_content_indices), sorted_content_length.cpu(), batch_first=True) # [batch_size * news_num, max_content_length, word_embedding_dim]
        # 2. selective LSTM encoding
        sorted_title_h, (sorted_title_h_n, sorted_title_c_n) = self.title_lstm(sorted_title)
        sorted_content_h, (sorted_content_h_n, sorted_content_c_n) = self.content_lstm(sorted_content)
        sorted_title_m = torch.cat([sorted_title_c_n[0], sorted_title_c_n[1]], dim=1)                                                         # [batch_size * news_num, hidden_dim * 2]
        sorted_content_m = torch.cat([sorted_content_c_n[0], sorted_content_c_n[1]], dim=1)                                                   # [batch_size * news_num, hidden_dim * 2]
        sorted_title_h, _ = pad_packed_sequence(sorted_title_h, batch_first=True, total_length=self.max_title_length)                         # [batch_size * news_num, max_title_length, hidden_dim * 2]
        sorted_content_h, _ = pad_packed_sequence(sorted_content_h, batch_first=True, total_length=self.max_content_length)                   # [batch_size * news_num, max_content_length, hidden_dim * 2]
        sorted_title_gate = torch.sigmoid(self.title_H(sorted_title_h) + self.title_M(sorted_content_m).unsqueeze(dim=1))                     # [batch_size * news_num, max_title_length, hidden_dim * 2]
        sorted_content_gate = torch.sigmoid(self.content_H(sorted_content_h) + self.content_M(sorted_title_m).unsqueeze(dim=1))               # [batch_size * news_num, max_content_length, hidden_dim * 2]
        title_h = (sorted_title_h * sorted_title_gate).index_select(0, desorted_title_indices)                                                # [batch_size * news_num, max_title_length, hidden_dim * 2]
        content_h = (sorted_content_h * sorted_content_gate).index_select(0, desorted_content_indices)                                        # [batch_size * news_num, max_content_length, hidden_dim * 2]
        # 3. self-attention
        title_self = self.title_self_attention(title_h, title_mask)                                                                           # [batch_size * news_num, hidden_dim * 2]
        content_self = self.content_self_attention(content_h, content_mask)                                                                   # [batch_size * news_num, hidden_dim * 2]
        news_representation = torch.cat([title_self, content_self], dim=1).view([batch_size, news_num, self.hidden_dim * 4])                  # [batch_size * news_num, hidden_dim * 4]
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                                 # [batch_size, news_num, news_embedding_dim]
        return news_representation


class SUE_wo_GCN(UserEncoder):
    def __init__(self, news_encoder, config):
        super(SUE_wo_GCN, self).__init__(news_encoder, config)
        self.attention_dim = max(config.attention_dim, self.news_embedding_dim // 4)
        self.intraCluster_K = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=True)
        self.intraCluster_Q = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=True)
        self.clusterFeatureAffine = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.interClusterAttention = ScaledDotProduct_CandidateAttention(self.news_embedding_dim, self.news_embedding_dim, self.attention_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.category_num = config.category_num + 1 # extra one category index for padding news
        self.max_history_num = config.max_history_num
        self.attention_scalar = math.sqrt(float(self.attention_dim))

    def initialize(self):
        nn.init.xavier_uniform_(self.intraCluster_K.weight)
        nn.init.zeros_(self.intraCluster_K.bias)
        nn.init.xavier_uniform_(self.intraCluster_Q.weight)
        nn.init.zeros_(self.intraCluster_Q.bias)
        nn.init.xavier_uniform_(self.clusterFeatureAffine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.clusterFeatureAffine.bias)
        self.interClusterAttention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)
        batch_news_num = batch_size * news_num
        user_history_category_mask[:, -1] = 1
        user_history_category_mask = user_history_category_mask.unsqueeze(dim=1).expand(-1, news_num, -1).contiguous()                         # [batch_size, news_num, category_num]
        user_history_category_indices = user_history_category_indices.unsqueeze(dim=1).expand(-1, news_num, -1)                                # [batch_size, news_num, max_history_num]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                                                 # [batch_size, max_history_num, news_embedding_dim]
        history_embedding = history_embedding.unsqueeze(dim=1).expand(-1, news_num, -1, -1)                                                    # [batch_size, news_num, max_history_num, news_embedding_dim]
        # 1. Intra-cluster attention
        K = self.intraCluster_K(history_embedding).view([batch_news_num, self.max_history_num, self.attention_dim])                            # [batch_size * news_num, max_history_num, attention_dim]
        Q = self.intraCluster_Q(candidate_news_representation).view([batch_news_num, self.attention_dim, 1])                                   # [batch_size * news_num, attention_dim, 1]
        a = torch.bmm(K, Q).view([batch_size, news_num, self.max_history_num]) / self.attention_scalar                                         # [batch_size, news_num, max_history_num]
        alpha_intra = scatter_softmax(a, user_history_category_indices, 2).unsqueeze(dim=3)                                                    # [batch_size, news_num, max_history_num, 1]
        intra_cluster_feature = scatter_sum(alpha_intra * history_embedding, user_history_category_indices, dim=2, dim_size=self.category_num) # [batch_size, news_num, category_num, news_embedding_dim]
        # perform non-linear transformation on intra-cluster features
        intra_cluster_feature = self.dropout(F.relu(self.clusterFeatureAffine(intra_cluster_feature), inplace=True) + intra_cluster_feature)   # [batch_size, news_num, category_num, news_embedding_dim]
        # 2. Inter-cluster attention
        inter_cluster_feature = self.interClusterAttention(
            intra_cluster_feature.view([batch_news_num, self.category_num, self.news_embedding_dim]),
            candidate_news_representation.view([batch_news_num, self.news_embedding_dim]),
            mask=user_history_category_mask.view([batch_news_num, self.category_num])
        ).view([batch_size, news_num, self.news_embedding_dim])                                                                                # [batch_size, news_num, news_embedding_dim]
        return inter_cluster_feature


class SUE_wo_HCA(UserEncoder):
    def __init__(self, news_encoder, config):
        super(SUE_wo_HCA, self).__init__(news_encoder, config)
        self.max_history_num = config.max_history_num
        self.proxy_node_embedding = nn.Parameter(torch.zeros([config.category_num, self.news_embedding_dim]))
        self.gcn = GCN(in_dim=self.news_embedding_dim, out_dim=self.news_embedding_dim, hidden_dim=self.news_embedding_dim, num_layers=config.gcn_layer_num, dropout=config.dropout_rate / 2, residual=not config.no_gcn_residual, layer_norm=config.gcn_layer_norm)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)

    def initialize(self):
        nn.init.zeros_(self.proxy_node_embedding)
        self.gcn.initialize()
        self.attention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, user_embedding) # [batch_size, max_history_num, news_embedding_dim]
        # 1. GCN
        history_embedding = torch.cat([history_embedding, self.dropout_(self.proxy_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1))], dim=1)   # [batch_size, max_history_num + category_num, news_embedding_dim]
        gcn_feature = self.gcn(history_embedding, user_history_graph) + history_embedding                                                                 # [batch_size, max_history_num + category_num, news_embedding_dim]
        gcn_feature = gcn_feature[:, :self.max_history_num, :]                                                                                            # [batch_size, max_history_num, news_embedding_dim]
        # 2. Plain attention
        user_representation = self.attention(gcn_feature).unsqueeze(dim=1).repeat(1, news_num, 1)                                                         # [batch_size, news_num, news_embedding_dim]
        return user_representation
