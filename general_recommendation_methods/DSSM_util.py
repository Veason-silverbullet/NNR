import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import random
import numpy as np
import json
import platform
from torch.utils.data import DataLoader
from evaluate import scoring


def transform_term_vectors(tfidf_dict, length):
    term_vector_dict = {}
    seq_len = {}
    for ID in tfidf_dict:
        tfidf = tfidf_dict[ID]
        tfidf_list = [[v, tfidf[v]] for v in tfidf]
        tfidf_list.sort(key=lambda x: x[1], reverse=True)
        term_indices = [0 for _ in range(length)]
        term_weights = [0 for _ in range(length)]
        len_length = min(len(tfidf_list), length)
        seq_len[ID] = np.float32(len_length) if len_length != 0 else np.float32(1e12)
        for i in range(len_length):
            term_indices[i] = tfidf_list[i][0]
            term_weights[i] = tfidf_list[i][1]
        term_vector_dict[ID] = [np.array(term_indices), np.array(term_weights, dtype=np.float32)]
    return term_vector_dict, seq_len


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description='DSSM')
        # General config
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test', 'debug'], help='Mode')
        parser.add_argument('--dev_model_path', type=str, default='best_model/DSSM/#1/DSSM', help='Dev model path')
        parser.add_argument('--test_model_path', type=str, default='best_model/DSSM/#1/DSSM', help='Test model path')
        parser.add_argument('--device_id', type=int, default=0, help='Device ID to run the model')
        parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
        # Dataset config
        parser.add_argument('--train_root', type=str, default='../../MIND/200000/train', help='Directory root of train data')
        parser.add_argument('--dev_root', type=str, default='../../MIND/200000/dev', help='Directory root of dev data')
        parser.add_argument('--test_root', type=str, default='../../MIND/200000/test', help='Directory root of test data')
        parser.add_argument('--news_word_num', type=int, default=200, help='Max word num in news sequence')
        parser.add_argument('--user_word_num', type=int, default=3200, help='Max word num in user sequence')
        # Training config
        parser.add_argument('--negative_sample_num', type=int, default=4, help='Negative sample number of each positive sample')
        parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=0, help='Optimizer weight decay')
        parser.add_argument('--gradient_clip_norm', type=float, default=4, help='Gradient clip norm (non-positive value for no clipping)')
        # Dev config
        parser.add_argument('--dev_criterion', type=str, default='auc', choices=['auc', 'mrr', 'ndcg', 'ndcg10'], help='Dev criterion to select model')
        parser.add_argument('--early_stopping_epoch', type=int, default=5, help='Epoch number of stop training after dev result does not improve')
        # Model config
        parser.add_argument('--word_embedding_dim', type=int, default=300, choices=[50, 100, 200, 300], help='Word embedding dimension')
        parser.add_argument('--DSSM_hidden_dim', type=int, default=512, help='DSSM hidden dimension')
        parser.add_argument('--DSSM_feature_dim', type=int, default=512, help='DSSM feature dimension')
        parser.add_argument('--dropout_rate', type=float, default=0, help='Dropout rate')

        args = parser.parse_args()
        self.mode = args.mode
        self.dev_model_path = args.dev_model_path
        self.test_model_path = args.test_model_path
        self.device_id = args.device_id
        self.seed = args.seed
        self.train_root = args.train_root
        self.dev_root = args.dev_root
        self.test_root = args.test_root
        self.news_word_num = args.news_word_num
        self.user_word_num = args.user_word_num
        self.negative_sample_num = args.negative_sample_num
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.gradient_clip_norm = args.gradient_clip_norm
        self.dev_criterion = args.dev_criterion
        self.early_stopping_epoch = args.early_stopping_epoch
        self.word_embedding_dim = args.word_embedding_dim
        self.DSSM_hidden_dim = args.DSSM_hidden_dim
        self.DSSM_feature_dim = args.DSSM_feature_dim
        self.dropout_rate = args.dropout_rate

        gpu_available = torch.cuda.is_available()
        assert gpu_available, 'GPU is not available'
        torch.cuda.set_device(self.device_id)
        torch.manual_seed(self.seed if self.seed >= 0 else (int)(time.time()))
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.benchmark = False # To make replicable
        torch.backends.cudnn.deterministic = True
        random.seed(self.seed)
        np.random.seed(self.seed)

        if not os.path.exists('./experiment_config'):
            os.mkdir('./experiment_config')
        if not os.path.exists('./experiment_config/DSSM'):
            os.mkdir('./experiment_config/DSSM')
        if not os.path.exists('./models'):
            os.mkdir('./models')
        if not os.path.exists('./models/DSSM'):
            os.mkdir('./models/DSSM')
        if not os.path.exists('./best_model'):
            os.mkdir('./best_model')
        if not os.path.exists('./best_model/DSSM'):
            os.mkdir('./best_model/DSSM')
        if not os.path.exists('./dev'):
            os.mkdir('./dev')
        if not os.path.exists('./dev/ref'):
            os.mkdir('./dev/ref')
        if not os.path.exists('./dev/res'):
            os.mkdir('./dev/res')
        if not os.path.exists('./dev/res/DSSM'):
            os.mkdir('./dev/res/DSSM')
        if not os.path.exists('./test'):
            os.mkdir('./test')
        if not os.path.exists('./test/ref'):
            os.mkdir('./test/ref')
        if not os.path.exists('./test/res'):
            os.mkdir('./test/res')
        if not os.path.exists('./test/res/DSSM'):
            os.mkdir('./test/res/DSSM')
        if not os.path.exists('./results'):
            os.mkdir('./results')
        if not os.path.exists('./results/DSSM'):
            os.mkdir('./results/DSSM')
        if not os.path.exists('./dev/ref/truth.txt'):
            with open(os.path.join(self.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_f:
                with open('./dev/ref/truth.txt', 'w', encoding='utf-8') as truth_f:
                    for dev_ID, line in enumerate(dev_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                        truth_f.write(('' if dev_ID == 0 else '\n') + str(dev_ID + 1) + ' ' + str(labels).replace(' ', ''))
        if not os.path.exists('./test/ref/truth.txt'):
            with open(os.path.join(self.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as test_f:
                with open('./test/ref/truth.txt', 'w', encoding='utf-8') as truth_f:
                    for test_ID, line in enumerate(test_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                        truth_f.write(('' if test_ID == 0 else '\n') + str(test_ID + 1) + ' ' + str(labels).replace(' ', ''))

        if not os.path.exists('news_term_vectors.pkl') or not os.path.exists('news_seq_len.pkl'):
            with open('news_tfidf.pkl', 'rb') as news_tfidf_f:
                news_tfidf_dict = pickle.load(news_tfidf_f)
                self.news_term_vectors, self.news_seq_len = transform_term_vectors(news_tfidf_dict, self.news_word_num)
                with open('news_term_vectors.pkl', 'wb') as news_term_vectors_f, open('news_seq_len.pkl', 'wb') as news_seq_len_f:
                    pickle.dump(self.news_term_vectors, news_term_vectors_f)
                    pickle.dump(self.news_seq_len, news_seq_len_f)
        else:
            with open('news_term_vectors.pkl', 'rb') as news_term_vectors_f, open('news_seq_len.pkl', 'rb') as news_seq_len_f:
                self.news_term_vectors = pickle.load(news_term_vectors_f)
                self.news_seq_len = pickle.load(news_seq_len_f)
        if not os.path.exists('user_term_vectors.pkl') or not os.path.exists('user_seq_len.pkl'):
            with open('user_tfidf.pkl', 'rb') as user_tfidf_f:
                user_tfidf_dict = pickle.load(user_tfidf_f)
                self.user_term_vectors, self.user_seq_len = transform_term_vectors(user_tfidf_dict, self.user_word_num)
                with open('user_term_vectors.pkl', 'wb') as user_term_vectors_f, open('user_seq_len.pkl', 'wb') as user_seq_len_f:
                    pickle.dump(self.user_term_vectors, user_term_vectors_f)
                    pickle.dump(self.user_seq_len, user_seq_len_f)
        else:
            with open('user_term_vectors.pkl', 'rb') as user_term_vectors_f, open('user_seq_len.pkl', 'rb') as user_seq_len_f:
                self.user_term_vectors = pickle.load(user_term_vectors_f)
                self.user_seq_len = pickle.load(user_seq_len_f)
        self.vocabulary_size = 0
        for term_vectors in self.news_term_vectors.values():
            self.vocabulary_size = max(self.vocabulary_size, max(term_vectors[0]))
        for term_vectors in self.user_term_vectors.values():
            self.vocabulary_size = max(self.vocabulary_size, max(term_vectors[0]))
        self.vocabulary_size += 1

    def print_log(self):
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        print('train_root :', self.train_root)
        print('seed :', self.seed)
        print('news_word_num :', self.news_word_num)
        print('user_word_num :', self.user_word_num)
        print('negative_sample_num :', self.negative_sample_num)
        print('epoch :', self.epoch)
        print('batch_size :', self.batch_size)
        print('lr :', self.lr)
        print('weight_decay :', self.weight_decay)
        print('gradient_clip_norm :', self.gradient_clip_norm)
        print('dev_criterion :', self.dev_criterion)
        print('early_stopping_epoch :', self.early_stopping_epoch)
        print('word_embedding_dim :', self.word_embedding_dim)
        print('DSSM_hidden_dim :', self.DSSM_hidden_dim)
        print('DSSM_feature_dim :', self.DSSM_feature_dim)
        print('dropout_rate :', self.dropout_rate)
        print('*' * 32 + ' Experiment setting ' + '*' * 32)

    def write_config_json(self, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump({
                'train_root': self.train_root,
                'seed': self.seed,
                'news_word_num': self.news_word_num,
                'user_word_num': self.user_word_num,
                'negative_sample_num': self.negative_sample_num,
                'epoch': self.epoch,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'gradient_clip_norm': self.gradient_clip_norm,
                'dev_criterion': self.dev_criterion,
                'early_stopping_epoch': self.early_stopping_epoch,
                'word_embedding_dim': self.word_embedding_dim,
                'DSSM_hidden_dim': self.DSSM_hidden_dim,
                'DSSM_feature_dim': self.DSSM_feature_dim,
                'dropout_rate': self.dropout_rate
            }, f)


def get_run_index():
    assert os.path.exists('./results/DSSM'), 'result directory does not exist'
    max_index = 0
    for result_file in os.listdir('./results/DSSM'):
        if result_file.strip()[0] == '#' and result_file.strip()[-4:] == '-dev':
            index = int(result_file.strip()[1:-4])
            max_index = max(index, max_index)
    with open('./results/DSSM/#' + str(max_index + 1) + '-dev', 'w', encoding='utf-8') as result_f:
        pass
    return max_index + 1


def compute_scores(model, dataset, batch_size, mode, result_file):
    assert mode in ['dev', 'test'], 'mode must be choosen from \'dev\' or \'test\''
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size // 8 if platform.system() == 'Linux' else 0)
    scores = torch.zeros([len(dataset.indices)]).cuda()
    index = 0
    model.eval()
    with torch.no_grad():
        for (user_indices, user_weights, user_seq_len, news_indices, news_weights, news_seq_len) in dataloader:
            user_indices = user_indices.cuda(non_blocking=True)
            user_weights = user_weights.cuda(non_blocking=True)
            user_seq_len = user_seq_len.cuda(non_blocking=True)
            news_indices = news_indices.cuda(non_blocking=True)
            news_weights = news_weights.cuda(non_blocking=True)
            news_seq_len = news_seq_len.cuda(non_blocking=True)
            batch_size = user_indices.size(0)
            scores[index: index+batch_size] = model(user_indices, user_weights, user_seq_len, news_indices.unsqueeze(dim=1), news_weights.unsqueeze(dim=1), news_seq_len.unsqueeze(dim=1)).squeeze(dim=1) # [batch_size]
            index += batch_size
    scores = scores.tolist()
    sub_scores = [[] for _ in range(dataset.indices[-1] + 1)]
    for i, index in enumerate(dataset.indices):
        sub_scores[index].append([scores[i], len(sub_scores[index])])
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    with open('./' + mode + '/ref/truth.txt', 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
        auc, mrr, ndcg, ndcg10 = scoring(truth_f, result_f)
    return auc, mrr, ndcg, ndcg10
