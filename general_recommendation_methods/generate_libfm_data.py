import os
import pickle
import argparse
import random
parser = argparse.ArgumentParser(description='Generate libfm data')
parser.add_argument('--dataset', type=str, default='200k', choices=['200k', 'small', 'large'], help='Dataset type')
parser.add_argument('--negative_sample_num', type=int, default=4, help='Negative sample number of each positive sample')
args = parser.parse_args()
dataset = args.dataset
train_root = '../../MIND-%s/train' % dataset
dev_root = '../../MIND-%s/dev' % dataset
test_root = '../../MIND-%s/test' % dataset
negative_sample_num = args.negative_sample_num


def tfidf2str(tfidf_dict, offset=0):
    str_dict = {}
    for ID in tfidf_dict:
        tfidf = tfidf_dict[ID]
        s = ''
        for index in tfidf:
            s += ' %d:%.12f' % (index + offset, tfidf[index])
        str_dict[ID] = s
    return str_dict

def generate_libfm_data():
    with open('offset-%s.txt' % dataset, 'r', encoding='utf-8') as offset_f:
        offset1 = int(offset_f.readline().strip())
        offset2 = int(offset_f.readline().strip())
        offset3 = int(offset_f.readline().strip())
    with open('news_ID-%s.pkl' % dataset, 'rb') as news_ID_f:
        news_ID_dict = pickle.load(news_ID_f)
    with open('user_ID-%s.pkl' % dataset, 'rb') as user_ID_f:
        user_ID_dict = pickle.load(user_ID_f)
    if not os.path.exists('news_tfidf_str-%s.pkl' % dataset):
        with open('news_tfidf-%s.pkl' % dataset, 'rb') as news_tfidf_f:
            news_tfidf = pickle.load(news_tfidf_f)
        news_tfidf_str = tfidf2str(news_tfidf, offset=offset1 + offset2)
        with open('news_tfidf_str-%s.pkl' % dataset, 'wb') as news_tfidf_str_f:
            pickle.dump(news_tfidf_str, news_tfidf_str_f)
    else:
        with open('news_tfidf_str-%s.pkl' % dataset, 'rb') as news_tfidf_str_f:
            news_tfidf_str = pickle.load(news_tfidf_str_f)
    if not os.path.exists('user_tfidf_str-%s.pkl' % dataset):
        with open('user_tfidf-%s.pkl' % dataset, 'rb') as user_tfidf_f:
            user_tfidf = pickle.load(user_tfidf_f)
        user_tfidf_str = tfidf2str(user_tfidf, offset=offset1 + offset2 + offset3)
        with open('user_tfidf_str-%s.pkl' % dataset, 'wb') as user_tfidf_str_f:
            pickle.dump(user_tfidf_str, user_tfidf_str_f)
    else:
        with open('user_tfidf_str-%s.pkl' % dataset, 'rb') as user_tfidf_str_f:
            user_tfidf_str = pickle.load(user_tfidf_str_f)
    with open('train-%s.libfm' % dataset, 'w', encoding='utf-8') as train_f:
        with open(os.path.join(train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
            for line in behaviors_f:
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                user_str = user_tfidf_str[user_ID]
                positive_samples = []
                negative_samples = []
                for impression in impressions.strip().split(' '):
                    if impression[-1] == '1':
                        positive_samples.append(impression[:-2])
                    else:
                        negative_samples.append(impression[:-2])
                positive_samples_num = len(positive_samples)
                negative_samples_num = len(negative_samples)
                if positive_samples_num * negative_sample_num >= negative_samples_num:
                    k = 0
                    for i in range(positive_samples_num):
                        train_f.write('1 %d:1 %d:1 %s %s\n' % (news_ID_dict[positive_samples[i]], user_ID_dict[user_ID] + offset1, news_tfidf_str[positive_samples[i]], user_str))
                        for j in range(negative_sample_num):
                            train_f.write('0 %d:1 %d:1 %s %s\n' % (news_ID_dict[negative_samples[k % negative_samples_num]], user_ID_dict[user_ID] + offset1, news_tfidf_str[negative_samples[k % negative_samples_num]], user_str))
                            k += 1
                else:
                    sample_index = random.sample(range(negative_samples_num), positive_samples_num * negative_sample_num)
                    k = 0
                    for i in range(positive_samples_num):
                        train_f.write('1 %d:1 %d:1 %s %s\n' % (news_ID_dict[positive_samples[i]], user_ID_dict[user_ID] + offset1, news_tfidf_str[positive_samples[i]], user_str))
                        for j in range(negative_sample_num):
                            train_f.write('0 %d:1 %d:1 %s %s\n' % (news_ID_dict[negative_samples[sample_index[k]]], user_ID_dict[user_ID] + offset1, news_tfidf_str[negative_samples[sample_index[k]]], user_str))
                            k += 1
    with open('dev-%s.libfm' % dataset, 'w', encoding='utf-8') as dev_f:
        with open(os.path.join(dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
            for line in behaviors_f:
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                user_str = user_tfidf_str[user_ID]
                for impression in impressions.strip().split(' '):
                    dev_f.write('%s %d:1 %d:1 %s %s\n' % (impression[-1], news_ID_dict[impression[:-2]], user_ID_dict[user_ID] + offset1, news_tfidf_str[impression[:-2]], user_str))
    with open('test-%s.libfm' % dataset, 'w', encoding='utf-8') as test_f:
        with open(os.path.join(test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
            for line in behaviors_f:
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                user_str = user_tfidf_str[user_ID]
                for impression in impressions.strip().split(' '):
                    test_f.write('%s %d:1 %d:1 %s %s\n' % (impression[-1], news_ID_dict[impression[:-2]], user_ID_dict[user_ID] + offset1, news_tfidf_str[impression[:-2]], user_str))


if __name__ == '__main__':
    generate_libfm_data()
