import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import random
import time
from tqdm import tqdm
import tensorflow as tf
from tensorflow import feature_column as fc
import pandas as pd
from functools import partial
from wide_deep_util import Config, get_run_index, input_func, evaluate
parser = argparse.ArgumentParser(description='Wide & deep')
parser.add_argument('--dataset', type=str, default='200k', choices=['200k', 'small', 'large'], help='Dataset type')
args = parser.parse_args()
dataset = args.dataset
USER_WORD_ID_TFIDF_FILE = 'user_word_ID_TFIDF-%s.pkl' % dataset
NEWS_WORD_ID_TFIDF_FILE = 'news_word_ID_TFIDF-%s.pkl' % dataset
DEV_DF_FILE = 'dev_df-%s.pkl' % dataset
TEST_DF_FILE = 'test_df-%s.pkl' % dataset
DEV_INDICES_FILE = 'dev_indices-%s.pkl' % dataset
TEST_INDICES_FILE = 'test_indices-%s.pkl' % dataset
tf.compat.v1.enable_eager_execution()


def read_data(config: Config):
    with open('user_ID-%s.pkl' % dataset, 'rb') as user_ID_f, open('news_ID-%s.pkl' % dataset, 'rb') as news_ID_f, open('user_tfidf-%s.pkl' % dataset, 'rb') as user_tfidf_dict_f, open('news_tfidf-%s.pkl' % dataset, 'rb') as news_tfidf_dict_f, open('offset-%s.txt' % dataset, 'r', encoding='utf-8') as offset_f:
        user_ID_dict = pickle.load(user_ID_f)
        news_ID_dict = pickle.load(news_ID_f)
        user_tfidf_dict = pickle.load(user_tfidf_dict_f)
        news_tfidf_dict = pickle.load(news_tfidf_dict_f)
        config.news_num = int(offset_f.readline().strip())
        config.user_num = int(offset_f.readline().strip())
        config.vocabulary_size = int(offset_f.readline().strip())
        config.word_num = 0
        for ID_TFIDF in user_tfidf_dict.values():
            config.word_num = max(config.word_num, len(ID_TFIDF))
        if not os.path.exists(USER_WORD_ID_TFIDF_FILE) or not os.path.exists(NEWS_WORD_ID_TFIDF_FILE):
            user_word_sparse_tensor_dict = {}
            news_word_sparse_tensor_dict = {}
            for ID in user_tfidf_dict:
                ID_TFIDF = user_tfidf_dict[ID]
                element_num = len(ID_TFIDF)
                if element_num > 0:
                    user_word_sparse_tensor_dict[user_ID_dict[ID]] = [[[0, i] for i in range(element_num)], list(ID_TFIDF.keys()), list(ID_TFIDF.values())]
                else:
                    user_word_sparse_tensor_dict[user_ID_dict[ID]] = [[[0, 0]], [config.word_num], [0]]
            for ID in news_tfidf_dict:
                ID_TFIDF = news_tfidf_dict[ID]
                element_num = len(ID_TFIDF)
                news_word_sparse_tensor_dict[news_ID_dict[ID]] = [[[0, i] for i in range(element_num)], list(ID_TFIDF.keys()), list(ID_TFIDF.values())]
            with open(USER_WORD_ID_TFIDF_FILE, 'wb') as user_word_ID_TFIDF_f, open(NEWS_WORD_ID_TFIDF_FILE, 'wb') as news_word_ID_TFIDF_f:
                pickle.dump(user_word_sparse_tensor_dict, user_word_ID_TFIDF_f)
                pickle.dump(news_word_sparse_tensor_dict, news_word_ID_TFIDF_f)
        else:
            with open(USER_WORD_ID_TFIDF_FILE, 'rb') as user_word_ID_TFIDF_f, open(NEWS_WORD_ID_TFIDF_FILE, 'rb') as news_word_ID_TFIDF_f:
                user_word_sparse_tensor_dict = pickle.load(user_word_ID_TFIDF_f)
                news_word_sparse_tensor_dict = pickle.load(news_word_ID_TFIDF_f)
        config.word_num += 1
    with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
        label_0 = tf.constant([0], dtype=tf.int64)
        label_1 = tf.constant([1], dtype=tf.int64)
        train_positive_samples = []
        train_negative_samples = []
        for line in behaviors_f:
            positive_samples = []
            negative_samples = []
            impression_ID, user_ID, time, history, impressions = line.split('\t')
            _user_ID = user_ID_dict[user_ID]
            for impression in impressions.strip().split(' '):
                news_ID = impression[:-2]
                _news_ID = news_ID_dict[news_ID]
                if impression[-1] == '0':
                    negative_samples.append([_user_ID, _news_ID, label_0])
                else:
                    positive_samples.append([_user_ID, _news_ID, label_1])
            train_positive_samples.append(positive_samples)
            train_negative_samples.append(negative_samples)
    if not os.path.exists(DEV_DF_FILE) or not os.path.exists(TEST_DF_FILE) or not os.path.exists(DEV_INDICES_FILE) or not os.path.exists(TEST_INDICES_FILE):
        dev_indices = []
        test_indices = []
        for i, behaviors_file in enumerate([os.path.join(config.dev_root, 'behaviors.tsv'), os.path.join(config.test_root, 'behaviors.tsv')]):
            with open(behaviors_file, 'r', encoding='utf-8') as behaviors_f:
                data = []
                for j, line in enumerate(behaviors_f):
                    impression_ID, user_ID, time, history, impressions = line.split('\t')
                    _user_ID = user_ID_dict[user_ID]
                    for impression in impressions.strip().split(' '):
                        news_ID = impression[:-2]
                        _news_ID = news_ID_dict[news_ID]
                        data.append([_user_ID, _news_ID])
                        if i == 0:
                            dev_indices.append(j)
                        else:
                            test_indices.append(j)
                if i == 0:
                    dev_df = pd.DataFrame(data=data, columns=['user_ID', 'news_ID'])
                    with open(DEV_DF_FILE, 'wb') as dev_df_f:
                        pickle.dump(dev_df, dev_df_f)
                    with open(DEV_INDICES_FILE, 'wb') as dev_indices_f:
                        pickle.dump(dev_indices, dev_indices_f)
                else:
                    test_df = pd.DataFrame(data=data, columns=['user_ID', 'news_ID'])
                    with open(TEST_DF_FILE, 'wb') as test_df_f:
                        pickle.dump(test_df, test_df_f)
                    with open(TEST_INDICES_FILE, 'wb') as test_indices_f:
                        pickle.dump(test_indices, test_indices_f)
    else:
        with open(DEV_DF_FILE, 'rb') as dev_df_f, open(TEST_DF_FILE, 'rb') as test_df_f, open(DEV_INDICES_FILE, 'rb') as dev_indices_f, open(TEST_INDICES_FILE, 'rb') as test_indices_f:
            dev_df = pickle.load(dev_df_f)
            test_df = pickle.load(test_df_f)
            dev_indices = pickle.load(dev_indices_f)
            test_indices = pickle.load(test_indices_f)
    tf.compat.v1.disable_eager_execution()
    wide_columns, deep_columns = build_feature_columns(config)
    return wide_columns, deep_columns, train_positive_samples, train_negative_samples, dev_df, test_df, user_word_sparse_tensor_dict, news_word_sparse_tensor_dict, dev_indices, test_indices

def negatvie_sampling(train_positive_samples, train_negative_samples, negative_sample_num):
    start_time = time.time()
    data = []
    for i, positive_samples in enumerate(train_positive_samples):
        negative_samples = train_negative_samples[i]
        positive_samples_num = len(positive_samples)
        negative_samples_num = len(negative_samples)
        sample_num = negative_sample_num * positive_samples_num
        data += positive_samples
        if sample_num > negative_samples_num:
            data += random.sample(negative_samples * (sample_num // negative_samples_num + 1), sample_num)
        else:
            data += random.sample(negative_samples, sample_num)
    train_df = pd.DataFrame(data=data, columns=['user_ID', 'news_ID', 'click_label'])
    end_time = time.time()
    print('End negative sampling, sample num :%d, used time : %.3fs' % (len(data), end_time - start_time))
    return train_df

def build_feature_columns(config: Config):
    user_ids = fc.categorical_column_with_identity('user_ID', num_buckets=config.user_num)
    news_ids = fc.categorical_column_with_identity('news_ID', num_buckets=config.news_num)
    user_word_tfidf = fc.weighted_categorical_column(fc.categorical_column_with_identity('user_word_ID', num_buckets=config.vocabulary_size), 'user_word_TFIDF', dtype=tf.dtypes.float32)
    news_word_tfidf = fc.weighted_categorical_column(fc.categorical_column_with_identity('news_word_ID', num_buckets=config.vocabulary_size), 'news_word_TFIDF', dtype=tf.dtypes.float32)
    # feed both channels with the same features
    wide_columns = [
        user_ids,
        news_ids,
        user_word_tfidf,
        news_word_tfidf,
        fc.crossed_column([user_ids, news_ids], hash_bucket_size=512*1024)
    ]
    deep_columns = [
        fc.embedding_column(categorical_column=user_ids, dimension=config.user_embedding_dim, max_norm=config.user_embedding_dim ** 0.5),
        fc.embedding_column(categorical_column=news_ids, dimension=config.news_embedding_dim, max_norm=config.news_embedding_dim ** 0.5),
        fc.embedding_column(categorical_column=user_word_tfidf, dimension=config.word_embedding_dim, max_norm=config.word_embedding_dim ** 0.5, combiner='sqrtn'),
        fc.embedding_column(categorical_column=news_word_tfidf, dimension=config.word_embedding_dim, max_norm=config.word_embedding_dim ** 0.5, combiner='sqrtn')
    ]
    return wide_columns, deep_columns

def build_model(config: Config, wide_columns: list, deep_columns: list):
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir='models/wide_deep/#' + str(config.run_index),
        config=tf.estimator.RunConfig(tf_random_seed=config.seed, log_step_count_steps=1000000000, save_checkpoints_steps=1000000000),
        linear_feature_columns=wide_columns,
        linear_optimizer=partial(tf.keras.optimizers.Adagrad, learning_rate=config.lr), # Adagrad for sparse-feature optimization
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=(config.wide_deep_hidden_dim, config.wide_deep_hidden_dim, config.wide_deep_hidden_dim),
        dnn_optimizer=partial(tf.keras.optimizers.Adagrad, learning_rate=config.lr), # Adagrad for sparse-feature optimization
        dnn_dropout=config.dropout_rate,
        n_classes=2,
        batch_norm=config.use_batch_norm,
        linear_sparse_combiner='sqrtn' # https://www.tensorflow.org/api_docs/python/tf/compat/v1/feature_column/linear_model
    )


if __name__ == '__main__':
    config = Config()
    wide_columns, deep_columns, train_positive_samples, train_negative_samples, dev_df, test_df, user_word_sparse_tensor_dict, news_word_sparse_tensor_dict, dev_indices, test_indices = read_data(config)
    with tf.device('/device:GPU:' + str(config.device_id)):
        config.run_index = get_run_index()
        config.write_config_json('configs/wide_deep/#' + str(config.run_index) + '.json')
        if not os.path.exists('dev/res/wide_deep/#' + str(config.run_index)):
            os.mkdir('dev/res/wide_deep/#' + str(config.run_index))
        if not os.path.exists('test/res/wide_deep/#' + str(config.run_index)):
            os.mkdir('test/res/wide_deep/#' + str(config.run_index))
        print('Running : wide_deep\t#' + str(config.run_index))
        config.print_log()
        wide_deep = build_model(config, wide_columns, deep_columns)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        best_result = 0
        best_dev_epoch = 0
        epoch_not_increase = 0
        test_auc = 0
        test_mrr = 0
        test_ndcg = 0
        test_ndcg10 = 0
        try:
            for e in tqdm(range(config.epoch)):
                train_df = negatvie_sampling(train_positive_samples, train_negative_samples, config.negative_sample_num)
                train_fn = input_func(
                    df=train_df,
                    user_word_sparse_tensor_dict=user_word_sparse_tensor_dict,
                    news_word_sparse_tensor_dict=news_word_sparse_tensor_dict,
                    sparse_tensor_shape=[1, config.word_num],
                    y_col='click_label',
                    batch_size=config.batch_size,
                    seed=config.seed
                )
                wide_deep.train(input_fn=train_fn)
                auc, mrr, ndcg, ndcg10 = evaluate(wide_deep, config, dev_df, dev_indices, user_word_sparse_tensor_dict, news_word_sparse_tensor_dict, 'dev')
                print('Epoch %d : dev done\nDev criterions' % (e + 1))
                print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}'.format(auc, mrr, ndcg, ndcg10))
                if config.dev_criterion == 'auc':
                    if e == 0 or auc >= best_result:
                        best_result = auc
                        best_dev_epoch = e + 1
                        epoch_not_increase = 0
                    else:
                        epoch_not_increase += 1
                elif config.dev_criterion == 'mrr':
                    if e == 0 or mrr >= best_result:
                        best_result = mrr
                        best_dev_epoch = e + 1
                        epoch_not_increase = 0
                    else:
                        epoch_not_increase += 1
                elif config.dev_criterion == 'ndcg':
                    if e == 0 or ndcg >= best_result:
                        best_result = ndcg
                        best_dev_epoch = e + 1
                        epoch_not_increase = 0
                    else:
                        epoch_not_increase += 1
                else:
                    if e == 0 or ndcg10 >= best_result:
                        best_result = ndcg10
                        best_dev_epoch = e + 1
                        epoch_not_increase = 0
                    else:
                        epoch_not_increase += 1
                print('Best epoch :', best_dev_epoch)
                print('Best ' + config.dev_criterion + ' : ' + str(best_result))
                if epoch_not_increase == 0:
                    test_auc, test_mrr, test_ndcg, test_ndcg10 = evaluate(wide_deep, config, test_df, test_indices, user_word_sparse_tensor_dict, news_word_sparse_tensor_dict, 'test')
                if epoch_not_increase > config.early_stopping_epoch:
                    print('Training : wide_deep #' + str(config.run_index) + ' completed\nDev criterions:')
                    print('Best ' + config.dev_criterion + ' : ' + str(best_result))
                    print('Test :\nAUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (test_auc, test_mrr, test_ndcg, test_ndcg10))
                    with open('results/wide_deep/#%d-test' % config.run_index, 'w', encoding='utf-8') as f:
                        f.write('#' + str(config.run_index) + '\t' + str(test_auc) + '\t' + str(test_mrr) + '\t' + str(test_ndcg) + '\t' + str(test_ndcg10) + '\n')
                    break
        except tf.estimator.NanLossDuringTrainingError:
            print('Train NAN')
