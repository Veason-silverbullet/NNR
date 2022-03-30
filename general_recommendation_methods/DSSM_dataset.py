import os
import time
import random
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
from DSSM_util import Config


class TF_IDF_Train_Dataset(data.Dataset):
    def __init__(self, config: Config):
        super(TF_IDF_Train_Dataset, self).__init__()
        self.K = config.negative_sample_num
        self.news_term_vectors = config.news_term_vectors
        self.seq_len = config.news_seq_len
        self.user_indices = []
        self.user_weights = []
        self.user_seq_len = []
        self.negative_sample_indices = []
        self.negative_samples = []
        with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
            for line in behaviors_f:
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                negative_samples = []
                for impression in impressions.strip().split(' '):
                    if impression[-2:] == '-1':
                        self.user_indices.append(config.user_term_vectors[user_ID][0])
                        self.user_weights.append(config.user_term_vectors[user_ID][1])
                        self.user_seq_len.append(config.user_seq_len[user_ID])
                        self.negative_sample_indices.append(len(self.negative_samples))
                    else:
                        negative_samples.append(impression[:-2])
                self.negative_samples.append(negative_samples)
        self.num = len(self.user_indices)
        self.user_indices = np.array(self.user_indices, dtype=np.int64)
        self.user_weights = np.array(self.user_weights, dtype=np.float32)
        self.news_indices = np.zeros([self.num, 1 + self.K, config.news_word_num], dtype=np.int64)
        self.news_weights = np.zeros([self.num, 1 + self.K, config.news_word_num], dtype=np.float32)
        self.news_seq_len = np.zeros([self.num, 1 + self.K], dtype=np.float32)
        with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
            index = 0
            for line in behaviors_f:
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                for impression in impressions.strip().split(' '):
                    if impression[-2:] == '-1':
                        news_ID = impression[:-2]
                        self.news_indices[index, 0, :] = self.news_term_vectors[impression[:-2]][0]
                        self.news_weights[index, 0, :] = self.news_term_vectors[impression[:-2]][1]
                        self.news_seq_len[index, 0] = self.seq_len[news_ID]
                        index += 1
        assert index == self.num, 'logical error'

    def negative_sampling(self):
        print('Begin negative sampling, training sample num : %d' % self.num)
        start_time = time.time()
        for i in range(self.num):
            negative_samples = self.negative_samples[self.negative_sample_indices[i]]
            if self.K > len(negative_samples):
                negative_news_IDs = random.sample(negative_samples * (self.K // len(negative_samples) + 1), self.K)
            else:
                negative_news_IDs = random.sample(negative_samples, self.K)
            for j in range(0, self.K):
                news_term_vectors = self.news_term_vectors[negative_news_IDs[j]]
                self.news_indices[i, j + 1, :] = news_term_vectors[0]
                self.news_weights[i, j + 1, :] = news_term_vectors[1]
                self.news_seq_len[i, j + 1] = self.seq_len[negative_news_IDs[j]]
        end_time = time.time()
        print('End negative sampling, used time : %.3fs' % (end_time - start_time))

    # user_indices : [user_word_num]
    # user_weights : [user_word_num]
    # user_seq_len : [1]
    # news_indices : [news_num, news_word_num]
    # news_weights : [news_num, news_word_num]
    # news_seq_len : [news_num]
    def __getitem__(self, index):
        return self.user_indices[index], self.user_weights[index], self.user_seq_len[index], self.news_indices[index], self.news_weights[index], self.news_seq_len[index]

    def __len__(self):
        return self.num


class TF_IDF_DevTest_Dataset(data.Dataset):
    def __init__(self, config: Config, mode: str):
        super(TF_IDF_DevTest_Dataset, self).__init__()
        assert mode.lower() in ['dev', 'test']
        dataset_root = config.dev_root if mode.lower() == 'dev' else config.test_root
        self.news_term_vectors = config.news_term_vectors
        self.num = 0
        with open(os.path.join(dataset_root, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
            for line in behaviors_f:
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                self.num += len(impressions.strip().split(' '))
        self.user_indices = np.zeros([self.num, config.user_word_num], dtype=np.int64)
        self.user_weights = np.zeros([self.num, config.user_word_num], dtype=np.float32)
        self.user_seq_len = [0 for _ in range(self.num)]
        self.news_indices = np.zeros([self.num, config.news_word_num], dtype=np.int64)
        self.news_weights = np.zeros([self.num, config.news_word_num], dtype=np.float32)
        self.news_seq_len = [0 for _ in range(self.num)]
        self.indices = [0 for _ in range(self.num)]
        index = 0
        with open(os.path.join(dataset_root, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
            for i, line in enumerate(behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                for impression in impressions.strip().split(' '):
                    news_ID = impression[:-2]
                    self.user_indices[index, :] = config.user_term_vectors[user_ID][0]
                    self.user_weights[index, :] = config.user_term_vectors[user_ID][1]
                    self.user_seq_len[index] = config.user_seq_len[user_ID]
                    self.news_indices[index, :] = self.news_term_vectors[news_ID][0]
                    self.news_weights[index, :] = self.news_term_vectors[news_ID][1]
                    self.news_seq_len[index] = config.news_seq_len[news_ID]
                    self.indices[index] = i
                    index += 1
        assert index == self.num, 'logical error'

    # user_indices : [user_word_num]
    # user_weights : [user_word_num]
    # user_seq_len : [1]
    # news_indices : [news_word_num]
    # news_weights : [news_word_num]
    # news_seq_len : [1]
    def __getitem__(self, index):
        return self.user_indices[index], self.user_weights[index], self.user_seq_len[index], self.news_indices[index], self.news_weights[index], self.news_seq_len[index]

    def __len__(self):
        return self.num


if __name__ == '__main__':
    config = Config()
    train_dataset = TF_IDF_Train_Dataset(config)
    train_dataset.negative_sampling()
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8)
    for (user_indices, user_weights, user_seq_len, news_indices, news_weights, news_seq_len) in train_dataloader:
        print('user_indices', user_indices.size(), user_indices.dtype)
        print('user_weights', user_weights.size(), user_weights.dtype)
        print('user_seq_len', user_seq_len.size(), user_seq_len.dtype)
        print('news_indices', news_indices.size(), news_indices.dtype)
        print('news_weights', news_weights.size(), news_weights.dtype)
        print('news_seq_len', news_seq_len.size(), news_seq_len.dtype)
        break
    dev_test_dataset = TF_IDF_DevTest_Dataset(config, mode='dev')
    dev_test_dataloader = DataLoader(dev_test_dataset, batch_size=64, shuffle=False, num_workers=8)
    for (user_indices, user_weights, user_seq_len, news_indices, news_weights, news_seq_len) in dev_test_dataloader:
        print('user_indices', user_indices.size(), user_indices.dtype)
        print('user_weights', user_weights.size(), user_weights.dtype)
        print('user_seq_len', user_seq_len.size(), user_seq_len.dtype)
        print('news_indices', news_indices.size(), news_indices.dtype)
        print('news_weights', news_weights.size(), news_weights.dtype)
        print('news_seq_len', news_seq_len.size(), news_seq_len.dtype)
        break
