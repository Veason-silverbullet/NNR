import os
import json
import argparse
import itertools
import random
import numpy as np
import tensorflow as tf
from tensorflow.sparse import SparseTensor
from evaluate import scoring


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description='wide_deep')
        # General config
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test', 'debug'], help='Mode')
        parser.add_argument('--dev_model_path', type=str, default='best_model/wide_deep/#1/wide_deep', help='Dev model path')
        parser.add_argument('--test_model_path', type=str, default='best_model/wide_deep/#1/wide_deep', help='Test model path')
        parser.add_argument('--device_id', type=int, default=0, help='Device ID to run the model')
        parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
        parser.add_argument('--gpu_use_rate', type=float, default=1, help='GPU use rate')
        # Dataset config
        parser.add_argument('--dataset', type=str, default='200k', choices=['200k', 'small', 'large'], help='Dataset type')
        # Training config
        parser.add_argument('--negative_sample_num', type=int, default=4, help='Negative sample number of each positive sample')
        parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
        parser.add_argument('--batch_size', type=int, default=320, help='Batch size')
        parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
        # Dev config
        parser.add_argument('--dev_criterion', type=str, default='auc', choices=['auc', 'mrr', 'ndcg', 'ndcg10'], help='Dev criterion to select model')
        parser.add_argument('--early_stopping_epoch', type=int, default=5, help='Epoch number of stop training after dev result does not improve')
        # Model config
        parser.add_argument('--user_embedding_dim', type=int, default=300, help='User embedding dimension')
        parser.add_argument('--news_embedding_dim', type=int, default=300, help='News embedding dimension')
        parser.add_argument('--word_embedding_dim', type=int, default=300, choices=[50, 100, 200, 300], help='Word embedding dimension')
        parser.add_argument('--wide_deep_hidden_dim', type=int, default=300, help='Wide_deep hidden dimension')
        parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
        parser.add_argument('--use_batch_norm', default=False, action='store_true', help='Whether use batch normalization')

        self.attribute_dict = dict(vars(parser.parse_args()))
        for attribute in self.attribute_dict:
            setattr(self, attribute, self.attribute_dict[attribute])
        self.train_root = '../../MIND-%s/train' % self.dataset
        self.dev_root = '../../MIND-%s/dev' % self.dataset
        self.test_root = '../../MIND-%s/test' % self.dataset
        self.seed = self.seed if self.seed >= 0 else (int)(time.time())

        tf.get_logger().setLevel('ERROR')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_id)
        gpu_available = tf.test.is_gpu_available()
        assert gpu_available, 'GPU is not available'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=self.gpu_use_rate)))
        random.seed(self.seed)
        np.random.seed(self.seed)

        if not os.path.exists('configs'):
            os.mkdir('configs')
        if not os.path.exists('configs/wide_deep'):
            os.mkdir('configs/wide_deep')
        if not os.path.exists('models'):
            os.mkdir('models')
        if not os.path.exists('models/wide_deep'):
            os.mkdir('models/wide_deep')
        if not os.path.exists('best_model'):
            os.mkdir('best_model')
        if not os.path.exists('best_model/wide_deep'):
            os.mkdir('best_model/wide_deep')
        if not os.path.exists('dev'):
            os.mkdir('dev')
        if not os.path.exists('dev/ref'):
            os.mkdir('dev/ref')
        if not os.path.exists('dev/res'):
            os.mkdir('dev/res')
        if not os.path.exists('dev/res/wide_deep'):
            os.mkdir('dev/res/wide_deep')
        if not os.path.exists('test'):
            os.mkdir('test')
        if not os.path.exists('test/ref'):
            os.mkdir('test/ref')
        if not os.path.exists('test/res'):
            os.mkdir('test/res')
        if not os.path.exists('test/res/wide_deep'):
            os.mkdir('test/res/wide_deep')
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists('results/wide_deep'):
            os.mkdir('results/wide_deep')
        if not os.path.exists('dev/ref/truth.txt'):
            with open(os.path.join(self.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_f:
                with open('dev/ref/truth.txt', 'w', encoding='utf-8') as truth_f:
                    for dev_ID, line in enumerate(dev_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                        truth_f.write(('' if dev_ID == 0 else '\n') + str(dev_ID + 1) + ' ' + str(labels).replace(' ', ''))
        if not os.path.exists('test/ref/truth.txt'):
            with open(os.path.join(self.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as test_f:
                with open('test/ref/truth.txt', 'w', encoding='utf-8') as truth_f:
                    for test_ID, line in enumerate(test_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                        truth_f.write(('' if test_ID == 0 else '\n') + str(test_ID + 1) + ' ' + str(labels).replace(' ', ''))

    def print_log(self):
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        for attribute in self.attribute_dict:
            print(attribute + ' : ' + str(getattr(self, attribute)))
        print('*' * 32 + ' Experiment setting ' + '*' * 32)

    def write_config_json(self, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump({
                'train_root': self.train_root,
                'seed': self.seed,
                'negative_sample_num': self.negative_sample_num,
                'epoch': self.epoch,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'dev_criterion': self.dev_criterion,
                'early_stopping_epoch': self.early_stopping_epoch,
                'user_embedding_dim': self.user_embedding_dim,
                'news_embedding_dim': self.news_embedding_dim,
                'word_embedding_dim': self.word_embedding_dim,
                'wide_deep_hidden_dim': self.wide_deep_hidden_dim,
                'dropout_rate': self.dropout_rate,
                'use_batch_norm': self.use_batch_norm
            }, f)


def get_run_index():
    assert os.path.exists('results/wide_deep'), 'result directory does not exist'
    max_index = 0
    for result_file in os.listdir('results/wide_deep'):
        if result_file.strip()[0] == '#' and result_file.strip()[-4:] == '-dev':
            index = int(result_file.strip()[1:-4])
            max_index = max(index, max_index)
    with open('results/wide_deep/#' + str(max_index + 1) + '-dev', 'w', encoding='utf-8') as result_f:
        pass
    return max_index + 1


def compute_scores(scores, indices, mode, result_file):
    assert mode in ['dev', 'test'], 'mode must be choosen from \'dev\' or \'test\''
    sub_scores = [[] for _ in range(indices[-1] + 1)]
    for i, index in enumerate(indices):
        sub_scores[index].append([scores[i], len(sub_scores[index])])
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    with open(mode + '/ref/truth.txt', 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
        auc, mrr, ndcg, ndcg10 = scoring(truth_f, result_f)
    return auc, mrr, ndcg, ndcg10


def input_func(df, user_word_sparse_tensor_dict, news_word_sparse_tensor_dict, sparse_tensor_shape, y_col=None, batch_size=128, seed=None):
    X_df = df.copy()
    y = X_df.pop(y_col).values if y_col is not None else None
    X = {}
    for col in X_df.columns:
        values = X_df[col].values
        if isinstance(values[0], (list, np.ndarray)):
            values = np.array([l for l in values], dtype=np.float32)
        X[col] = values
    return lambda: _dataset(x=X, user_word_sparse_tensor_dict=user_word_sparse_tensor_dict, news_word_sparse_tensor_dict=news_word_sparse_tensor_dict, sparse_tensor_shape=sparse_tensor_shape, y=y, batch_size=batch_size, seed=seed)


def _dataset(x, user_word_sparse_tensor_dict, news_word_sparse_tensor_dict, sparse_tensor_shape, y=None, batch_size=128, seed=None):
    if y is not None:
        def train_dataset_generator():
            user_ids = x['user_ID']
            news_ids = x['news_ID']
            for i in range(len(user_ids)):
                user_ID = user_ids[i]
                news_ID = news_ids[i]
                user_indices, user_values1, user_values2 = user_word_sparse_tensor_dict[user_ID]
                news_indices, news_values1, news_values2 = news_word_sparse_tensor_dict[news_ID]
                yield user_ID, news_ID, user_indices, user_values1, user_values2, news_indices, news_values1, news_values2, y[i]
        dataset = tf.data.Dataset.from_generator(train_dataset_generator, (tf.int64, tf.int64, tf.int64, tf.int64, tf.float32, tf.int64, tf.int64, tf.float32, tf.float32))
        def train_dataset_map_func(user_ID, news_ID, user_indices, user_values1, user_values2, news_indices, news_values1, news_values2, y):
            X = {
                'user_ID': user_ID,
                'news_ID': news_ID,
                'user_word_ID': SparseTensor(indices=user_indices, values=user_values1, dense_shape=sparse_tensor_shape),
                'user_word_TFIDF': SparseTensor(indices=user_indices, values=user_values2, dense_shape=sparse_tensor_shape),
                'news_word_ID': SparseTensor(indices=news_indices, values=news_values1, dense_shape=sparse_tensor_shape),
                'news_word_TFIDF': SparseTensor(indices=news_indices, values=news_values2, dense_shape=sparse_tensor_shape)
            }
            return (X, y)
        dataset = dataset.map(train_dataset_map_func, num_parallel_calls=16)
        dataset = dataset.shuffle(1024, seed=seed)
    else:
        def eval_dataset_generator():
            user_ids = x['user_ID']
            news_ids = x['news_ID']
            for i in range(len(user_ids)):
                user_ID = user_ids[i]
                news_ID = news_ids[i]
                user_indices, user_values1, user_values2 = user_word_sparse_tensor_dict[user_ID]
                news_indices, news_values1, news_values2 = news_word_sparse_tensor_dict[news_ID]
                yield user_ID, news_ID, user_indices, user_values1, user_values2, news_indices, news_values1, news_values2
        dataset = tf.data.Dataset.from_generator(eval_dataset_generator, (tf.int64, tf.int64, tf.int64, tf.int64, tf.float32, tf.int64, tf.int64, tf.float32))
        def eval_dataset_map_func(user_ID, news_ID, user_indices, user_values1, user_values2, news_indices, news_values1, news_values2):
            X = {
                'user_ID': user_ID,
                'news_ID': news_ID,
                'user_word_ID': SparseTensor(indices=user_indices, values=user_values1, dense_shape=sparse_tensor_shape),
                'user_word_TFIDF': SparseTensor(indices=user_indices, values=user_values2, dense_shape=sparse_tensor_shape),
                'news_word_ID': SparseTensor(indices=news_indices, values=news_values1, dense_shape=sparse_tensor_shape),
                'news_word_TFIDF': SparseTensor(indices=news_indices, values=news_values2, dense_shape=sparse_tensor_shape)
            }
            return X
        dataset = dataset.map(eval_dataset_map_func, num_parallel_calls=16)
    return dataset.batch(batch_size)


def evaluate(model, config, eval_df, eval_indices, user_word_sparse_tensor_dict, news_word_sparse_tensor_dict, mode):
    assert mode in ['dev', 'test']
    predictions = list(
        itertools.islice(
            model.predict(
                input_fn=input_func(
                    df=eval_df,
                    user_word_sparse_tensor_dict=user_word_sparse_tensor_dict,
                    news_word_sparse_tensor_dict=news_word_sparse_tensor_dict,
                    sparse_tensor_shape=[1, config.word_num],
                    batch_size=config.batch_size * 2
                ), predict_keys=['logistic']
            ), len(eval_df)
        )
    )
    scores = [float(prediction['logistic'][0]) for prediction in predictions]
    assert len(scores) == len(eval_indices), 'logical error'
    auc, mrr, ndcg, ndcg10 = compute_scores(scores, eval_indices, mode, mode + '/res/wide_deep/#' + str(config.run_index) + '/wide_deep-' + str(config.run_index))
    return auc, mrr, ndcg, ndcg10
