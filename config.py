import os
import argparse
import time
import torch
import random
import numpy as np
import json
from prepare_MIND_dataset import prepare_MIND_200k, prepare_MIND_small, prepare_MIND_large


class Config:
    def parse_argument(self):
        parser = argparse.ArgumentParser(description='Neural news recommendation')
        # General config
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'], help='Mode')
        parser.add_argument('--news_encoder', type=str, default='CNE', choices=['CNE', 'CNN', 'MHSA', 'KCNN', 'HDC', 'NAML', 'PNE', 'DAE', 'Inception', 'NAML_Title', 'NAML_Content', 'CNE_Title', 'CNE_Content', 'CNE_wo_CS', 'CNE_wo_CA'], help='News encoder')
        parser.add_argument('--user_encoder', type=str, default='SUE', choices=['SUE', 'LSTUR', 'MHSA', 'ATT', 'CATT', 'FIM', 'PUE', 'GRU', 'OMAP', 'SUE_wo_GCN', 'SUE_wo_HCA'], help='User encoder')
        parser.add_argument('--dev_model_path', type=str, default='', help='Dev model path')
        parser.add_argument('--test_model_path', type=str, default='', help='Test model path')
        parser.add_argument('--test_output_file', type=str, default='', help='Specific test output file')
        parser.add_argument('--device_id', type=int, default=0, help='Device ID of GPU')
        parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
        parser.add_argument('--config_file', type=str, default='', help='Config file path')
        # Dataset config
        parser.add_argument('--dataset', type=str, default='200k', choices=['200k', 'small', 'large'], help='Dataset type')
        parser.add_argument('--tokenizer', type=str, default='MIND', choices=['MIND', 'NLTK'], help='Sentence tokenizer')
        parser.add_argument('--word_threshold', type=int, default=3, help='Word threshold')
        parser.add_argument('--max_title_length', type=int, default=32, help='Sentence truncate length for title')
        parser.add_argument('--max_abstract_length', type=int, default=128, help='Sentence truncate length for abstract')
        # Training config
        parser.add_argument('--negative_sample_num', type=int, default=4, help='Negative sample number of each positive sample')
        parser.add_argument('--max_history_num', type=int, default=50, help='Maximum number of history news for each user')
        parser.add_argument('--epoch', type=int, default=16, help='Training epoch')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=0, help='Optimizer weight decay')
        parser.add_argument('--gradient_clip_norm', type=float, default=4, help='Gradient clip norm (non-positive value for no clipping)')
        parser.add_argument('--world_size', type=int, default=1, help='World size of multi-process GPU training')
        # Dev config
        parser.add_argument('--dev_criterion', type=str, default='avg', choices=['auc', 'mrr', 'ndcg5', 'ndcg10', 'avg'], help='Validation criterion to select model')
        parser.add_argument('--early_stopping_epoch', type=int, default=5, help='Epoch number of stop training after dev result does not improve')
        # Model config
        parser.add_argument('--word_embedding_dim', type=int, default=300, choices=[50, 100, 200, 300], help='Word embedding dimension')
        parser.add_argument('--entity_embedding_dim', type=int, default=100, choices=[100], help='Entity embedding dimension')
        parser.add_argument('--context_embedding_dim', type=int, default=100, choices=[100], help='Context embedding dimension')
        parser.add_argument('--cnn_method', type=str, default='naive', choices=['naive', 'group3', 'group4', 'group5'], help='CNN group')
        parser.add_argument('--cnn_kernel_num', type=int, default=400, help='Number of CNN kernel')
        parser.add_argument('--cnn_window_size', type=int, default=3, help='Window size of CNN kernel')
        parser.add_argument('--attention_dim', type=int, default=200, help="Attention dimension")
        parser.add_argument('--head_num', type=int, default=20, help='Head number of multi-head self-attention')
        parser.add_argument('--head_dim', type=int, default=20, help='Head dimension of multi-head self-attention')
        parser.add_argument('--user_embedding_dim', type=int, default=50, help='User embedding dimension')
        parser.add_argument('--category_embedding_dim', type=int, default=50, help='Category embedding dimension')
        parser.add_argument('--subCategory_embedding_dim', type=int, default=50, help='SubCategory embedding dimension')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
        parser.add_argument('--no_self_connection', default=False, action='store_true', help='Whether the graph contains self-connection')
        parser.add_argument('--no_adjacent_normalization', default=False, action='store_true', help='Whether normalize the adjacent matrix')
        parser.add_argument('--gcn_normalization_type', type=str, default='symmetric', choices=['symmetric', 'asymmetric'], help='GCN normalization for adjacent matrix A (\"symmetric\" for D^{-\\frac{1}{2}}AD^{-\\frac{1}{2}}; \"asymmetric\" for D^{-\\frac{1}{2}}A)')
        parser.add_argument('--gcn_layer_num', type=int, default=4, help='Number of GCN layer')
        parser.add_argument('--no_gcn_residual', default=False, action='store_true', help='Whether apply residual connection to GCN')
        parser.add_argument('--gcn_layer_norm', default=False, action='store_true', help='Whether apply layer normalization to GCN')
        parser.add_argument('--hidden_dim', type=int, default=200, help='Encoder hidden dimension')
        parser.add_argument('--Alpha', type=float, default=0.1, help='Reconstruction loss weight for DAE')
        parser.add_argument('--long_term_masking_probability', type=float, default=0.1, help='Probability of masking long-term representation for LSTUR')
        parser.add_argument('--personalized_embedding_dim', type=int, default=200, help='Personalized embedding dimension for NPA')
        parser.add_argument('--HDC_window_size', type=int, default=3, help='Convolution window size of HDC for FIM')
        parser.add_argument('--HDC_filter_num', type=int, default=150, help='Convolution filter num of HDC for FIM')
        parser.add_argument('--conv3D_filter_num_first', type=int, default=32, help='3D matching convolution filter num of the first layer for FIM ')
        parser.add_argument('--conv3D_kernel_size_first', type=int, default=3, help='3D matching convolution kernel size of the first layer for FIM')
        parser.add_argument('--conv3D_filter_num_second', type=int, default=16, help='3D matching convolution filter num of the second layer for FIM ')
        parser.add_argument('--conv3D_kernel_size_second', type=int, default=3, help='3D matching convolution kernel size of the second layer for FIM')
        parser.add_argument('--maxpooling3D_size', type=int, default=3, help='3D matching pooling size for FIM ')
        parser.add_argument('--maxpooling3D_stride', type=int, default=3, help='3D matching pooling stride for FIM')
        parser.add_argument('--OMAP_head_num', type=int, default=3, help='Head num of OMAP for Hi-Fi Ark')
        parser.add_argument('--HiFi_Ark_regularizer_coefficient', type=float, default=0.1, help='Coefficient of regularization loss for Hi-Fi Ark')
        parser.add_argument('--click_predictor', type=str, default='dot_product', choices=['dot_product', 'mlp', 'sigmoid', 'FIM'], help='Click predictor')

        self.attribute_dict = dict(vars(parser.parse_args()))
        for attribute in self.attribute_dict:
            setattr(self, attribute, self.attribute_dict[attribute])
        self.train_root = '../MIND-%s/train' % self.dataset
        self.dev_root = '../MIND-%s/dev' % self.dataset
        self.test_root = '../MIND-%s/test' % self.dataset
        if self.dataset == 'small': # suggested configuration for MIND-small
            self.dropout_rate = 0.25
            self.gcn_layer_num = 3
        elif self.dataset == '200k': # suggested configuration for MIND-200k
            self.dropout_rate = 0.2
            self.gcn_layer_num = 4
            self.epoch = 8
        else: # suggested configuration for MIND-large
            self.dropout_rate = 0.1
            self.gcn_layer_num = 4
            self.epoch = 6
        self.seed = self.seed if self.seed >= 0 else (int)(time.time())
        self.attribute_dict['dropout_rate'] = self.dropout_rate
        self.attribute_dict['gcn_layer_num'] = self.gcn_layer_num
        self.attribute_dict['epoch'] = self.epoch
        self.attribute_dict['seed'] = self.seed
        if self.config_file != '':
            if os.path.exists(self.config_file):
                print('Get experiment settings from the config file : ' + self.config_file)
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
                    for attribute in self.attribute_dict:
                        if attribute in configs:
                            setattr(self, attribute, configs[attribute])
                            self.attribute_dict[attribute] = configs[attribute]
            else:
                raise Exception('Config file does not exist : ' + self.config_file)
        assert not (self.no_self_connection and not self.no_adjacent_normalization), 'Adjacent normalization of graph only can be set in case of self-connection'
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        for attribute in self.attribute_dict:
            print(attribute + ' : ' + str(getattr(self, attribute)))
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        assert self.batch_size % self.world_size == 0, 'For multi-gpu training, batch size must be divisible by world size'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '1024'


    def set_cuda(self):
        gpu_available = torch.cuda.is_available()
        assert gpu_available, 'GPU is not available'
        torch.cuda.set_device(self.device_id)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True # For reproducibility (https://pytorch.org/docs/stable/notes/randomness.html)


    def preliminary_setup(self):
        dataset_files = [
            self.train_root + '/news.tsv', self.train_root + '/behaviors.tsv', self.train_root + '/entity_embedding.vec', self.train_root + '/context_embedding.vec', 
            self.dev_root + '/news.tsv', self.dev_root + '/behaviors.tsv', self.dev_root + '/entity_embedding.vec', self.dev_root + '/context_embedding.vec', 
            self.test_root + '/news.tsv', self.test_root + '/behaviors.tsv', self.test_root + '/entity_embedding.vec', self.test_root + '/context_embedding.vec'
        ]
        if not all(list(map(os.path.exists, dataset_files))):
            exec('prepare_MIND_%s()' % self.dataset)

        model_name = self.news_encoder + '-' + self.user_encoder
        mkdirs = lambda x: os.makedirs(x) if not os.path.exists(x) else None
        self.config_dir = 'configs/' + self.dataset + '/' + model_name
        self.model_dir = 'models/' + self.dataset + '/' + model_name
        self.best_model_dir = 'best_model/' + self.dataset + '/' + model_name
        self.dev_res_dir = 'dev/res/' + self.dataset + '/' + model_name
        self.test_res_dir = 'test/res/' + self.dataset + '/' + model_name
        self.result_dir = 'results/' + self.dataset + '/' + model_name
        mkdirs(self.config_dir)
        mkdirs(self.model_dir)
        mkdirs(self.best_model_dir)
        mkdirs('dev/ref')
        mkdirs(self.dev_res_dir)
        mkdirs('test/ref')
        mkdirs(self.test_res_dir)
        mkdirs(self.result_dir)
        if not os.path.exists('dev/ref/truth-%s.txt' % self.dataset):
            with open(os.path.join(self.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_f:
                with open('dev/ref/truth-%s.txt' % self.dataset, 'w', encoding='utf-8') as truth_f:
                    for dev_ID, line in enumerate(dev_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                        truth_f.write(('' if dev_ID == 0 else '\n') + str(dev_ID + 1) + ' ' + str(labels).replace(' ', ''))
        if self.dataset != 'large':
            if not os.path.exists('test/ref/truth-%s.txt' % self.dataset):
                with open(os.path.join(self.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as test_f:
                    with open('test/ref/truth-%s.txt' % self.dataset, 'w', encoding='utf-8') as truth_f:
                        for test_ID, line in enumerate(test_f):
                            impression_ID, user_ID, time, history, impressions = line.split('\t')
                            labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                            truth_f.write(('' if test_ID == 0 else '\n') + str(test_ID + 1) + ' ' + str(labels).replace(' ', ''))
        else:
            self.prediction_dir = 'prediction/large/' + model_name
            mkdirs(self.prediction_dir)


    def __init__(self):
        self.parse_argument()
        self.preliminary_setup()
        self.set_cuda()


if __name__ == '__main__':
    config = Config()
