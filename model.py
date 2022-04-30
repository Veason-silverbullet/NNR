from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import newsEncoders
import userEncoders
import variantEncoders


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        # For main experiments of news encoding
        if config.news_encoder == 'CNE':
            self.news_encoder = newsEncoders.CNE(config)
        elif config.news_encoder == 'CNN':
            self.news_encoder = newsEncoders.CNN(config)
        elif config.news_encoder == 'MHSA':
            self.news_encoder = newsEncoders.MHSA(config)
        elif config.news_encoder == 'KCNN':
            self.news_encoder = newsEncoders.KCNN(config)
        elif config.news_encoder == 'HDC':
            self.news_encoder = newsEncoders.HDC(config)
        elif config.news_encoder == 'NAML':
            self.news_encoder = newsEncoders.NAML(config)
        elif config.news_encoder == 'PNE':
            self.news_encoder = newsEncoders.PNE(config)
        elif config.news_encoder == 'DAE':
            self.news_encoder = newsEncoders.DAE(config)
        elif config.news_encoder == 'Inception':
            self.news_encoder = newsEncoders.Inception(config)
        # For ablations of news encoding
        elif config.news_encoder == 'NAML_Title':
            self.news_encoder = variantEncoders.NAML_Title(config)
        elif config.news_encoder == 'NAML_Content':
            self.news_encoder = variantEncoders.NAML_Content(config)
        elif config.news_encoder == 'CNE_Title':
            self.news_encoder = variantEncoders.CNE_Title(config)
        elif config.news_encoder == 'CNE_Content':
            self.news_encoder = variantEncoders.CNE_Content(config)
        elif config.news_encoder == 'CNE_wo_CS':
            self.news_encoder = variantEncoders.CNE_wo_CS(config)
        elif config.news_encoder == 'CNE_wo_CA':
            self.news_encoder = variantEncoders.CNE_wo_CA(config)
        else:
            raise Exception(config.news_encoder + 'is not implemented')

        # For main experiments of user encoding
        if config.user_encoder == 'SUE':
            self.user_encoder = userEncoders.SUE(self.news_encoder, config)
        elif config.user_encoder == 'LSTUR':
            self.user_encoder = userEncoders.LSTUR(self.news_encoder, config)
        elif config.user_encoder == 'MHSA':
            self.user_encoder = userEncoders.MHSA(self.news_encoder, config)
        elif config.user_encoder == 'ATT':
            self.user_encoder = userEncoders.ATT(self.news_encoder, config)
        elif config.user_encoder == 'CATT':
            self.user_encoder = userEncoders.CATT(self.news_encoder, config)
        elif config.user_encoder == 'FIM':
            self.user_encoder = userEncoders.FIM(self.news_encoder, config)
        elif config.user_encoder == 'PUE':
            self.user_encoder = userEncoders.PUE(self.news_encoder, config)
        elif config.user_encoder == 'GRU':
            self.user_encoder = userEncoders.GRU(self.news_encoder, config)
        elif config.user_encoder == 'OMAP':
            self.user_encoder = userEncoders.OMAP(self.news_encoder, config)
        # For ablations of user encoding
        elif config.user_encoder == 'SUE_wo_GCN':
            self.user_encoder = variantEncoders.SUE_wo_GCN(self.news_encoder, config)
        elif config.user_encoder == 'SUE_wo_HCA':
            self.user_encoder = variantEncoders.SUE_wo_HCA(self.news_encoder, config)
        else:
            raise Exception(config.user_encoder + 'is not implemented')

        self.model_name = config.news_encoder + '-' + config.user_encoder
        self.news_embedding_dim = self.news_encoder.news_embedding_dim
        self.dropout = nn.Dropout(p=config.dropout_rate)
        if config.user_encoder == 'LSTUR':
            self.user_embedding = nn.Embedding(num_embeddings=config.user_num, embedding_dim=self.news_embedding_dim)
            self.use_user_embedding = True
        elif config.news_encoder == 'PNE' or config.user_encoder == 'PUE':
            self.user_embedding = nn.Embedding(num_embeddings=config.user_num, embedding_dim=config.user_embedding_dim)
            self.use_user_embedding = True
        else:
            self.use_user_embedding = False
        if config.news_encoder == 'HDC' or config.user_encoder == 'FIM':
            assert config.news_encoder == 'HDC' and config.user_encoder == 'FIM', 'HDC and FIM must be paired and can not be used alone'
            assert config.click_predictor == 'FIM', 'For the model FIM, the click predictor must be specially set as \'FIM\''
        self.click_predictor = config.click_predictor
        if self.click_predictor == 'mlp':
            self.mlp = nn.Linear(in_features=self.news_embedding_dim * 2, out_features=self.news_embedding_dim // 2, bias=True)
            self.out = nn.Linear(in_features=self.news_embedding_dim // 2, out_features=1, bias=True)
        elif self.click_predictor == 'FIM':
            # compute the output size of 3D convolution and pooling
            def compute_convolution_pooling_output_size(input_size):
                conv1_size = input_size - config.conv3D_kernel_size_first + 1
                pool1_size = (conv1_size - config.maxpooling3D_size) // config.maxpooling3D_stride + 1
                conv2_size = pool1_size - config.conv3D_kernel_size_second + 1
                pool2_size = (conv2_size - config.maxpooling3D_size) // config.maxpooling3D_stride + 1
                return pool2_size
            feature_size = compute_convolution_pooling_output_size(self.news_encoder.HDC_sequence_length) * \
                           compute_convolution_pooling_output_size(self.news_encoder.HDC_sequence_length) * \
                           compute_convolution_pooling_output_size(config.max_history_num) * \
                           config.conv3D_filter_num_second
            self.fc = nn.Linear(in_features=feature_size, out_features=1, bias=True)

    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()
        if self.use_user_embedding:
            nn.init.uniform_(self.user_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.user_embedding.weight[0])
        if self.click_predictor == 'mlp':
            nn.init.xavier_uniform_(self.mlp.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.mlp.bias)
        elif self.click_predictor == 'FIM':
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                      news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity):
        user_embedding = self.dropout(self.user_embedding(user_ID)) if self.use_user_embedding else None                                                                                                         # [batch_size, news_embedding_dim]
        news_representation = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, user_embedding) # [batch_size, 1 + negative_sample_num, news_embedding_dim]
        user_representation = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, news_representation)                           # [batch_size, 1 + negative_sample_num, news_embedding_dim]
        if self.click_predictor == 'dot_product':
            logits = (user_representation * news_representation).sum(dim=2) # dot-product
        elif self.click_predictor == 'mlp':
            context = self.dropout(F.relu(self.mlp(torch.cat([user_representation, news_representation], dim=2)), inplace=True))
            logits = self.out(context).squeeze(dim=2)
        elif self.click_predictor == 'FIM':
            logits = self.fc(user_representation).squeeze(dim=2)
        return logits
