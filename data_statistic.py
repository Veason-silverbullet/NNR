import os
import json
import collections
import re
pat = re.compile(r"[\w]+|[.,!?;|]")


# MIND tokenizer
def word_tokenize(s):
    return pat.findall(s.lower())


def news_statistic(train_root, dev_root):
    user_dict = {}
    news_dict = {}
    word_dict = {}
    word_counter = collections.Counter()
    train_cnt, train_max_title_len, train_title_len, train_max_content_len, train_content_len = 0, 0, 0, 0, 0
    dev_cnt, dev_max_title_len, dev_title_len, dev_max_content_len, dev_content_len = 0, 0, 0, 0, 0
    title_lens, content_lens = {}, {}
    train_title_length_counter = [0 for _ in range(128)]
    train_content_length_counter = [0 for _ in range(640)]
    dev_title_length_counter = [0 for _ in range(128)]
    dev_content_length_counter = [0 for _ in range(640)]
    with open(os.path.join(train_root, 'news.tsv'), 'r', encoding='utf-8') as train_news_file:
        for line in train_news_file:
            news_ID, category, subCategory, title, content, url, title_entities, content_entities = line.split('\t')
            title = list(map(lambda word: word.lower(), word_tokenize(title)))
            content = list(map(lambda word: word.lower(), word_tokenize(content)))
            for word in title:
                word_counter[word] += 1
            for word in content:
                word_counter[word] += 1
            if news_ID not in news_dict:
                news_dict[news_ID] = len(news_dict)
            train_max_title_len = max(len(title), train_max_title_len)
            train_title_len += len(title)
            title_lens[news_ID] = len(title)
            train_max_content_len = max(len(content), train_max_content_len)
            train_content_len += len(content)
            content_lens[news_ID] = len(content)
            train_title_length_counter[len(title)] += 1
            train_content_length_counter[len(content)] += 1
            train_cnt += 1
    with open(os.path.join(dev_root, 'news.tsv'), 'r', encoding='utf-8') as dev_news_file:
        for line in dev_news_file:
            news_ID, category, subCategory, title, content, url, title_entities, content_entities = line.split('\t')
            title = list(map(lambda word: word.lower(), word_tokenize(title)))
            content = list(map(lambda word: word.lower(), word_tokenize(content)))
            for word in title:
                word_counter[word] += 1
            for word in content:
                word_counter[word] += 1
            if news_ID not in news_dict:
                news_dict[news_ID] = len(news_dict)
            dev_max_title_len = max(len(title), dev_max_title_len)
            dev_title_len += len(title)
            title_lens[news_ID] = len(title)
            dev_max_content_len = max(len(content), dev_max_content_len)
            dev_content_len += len(content)
            content_lens[news_ID] = len(content)
            dev_title_length_counter[len(title)] += 1
            dev_content_length_counter[len(content)] += 1
            dev_cnt += 1
    train_title_length_accumulate = [0 for _ in range(128)]
    train_content_length_accumulate = [0 for _ in range(640)]
    dev_title_length_accumulate = [0 for _ in range(128)]
    dev_content_length_accumulate = [0 for _ in range(640)]
    train_title_length_accumulate[0] = train_title_length_counter[0]
    train_content_length_accumulate[0] = train_content_length_counter[0]
    dev_title_length_accumulate[0] = dev_title_length_counter[0]
    dev_content_length_accumulate[0] = dev_content_length_counter[0]
    for i in range(1, 128):
        train_title_length_accumulate[i] = train_title_length_accumulate[i - 1] + train_title_length_counter[i]
        dev_title_length_accumulate[i] = dev_title_length_accumulate[i - 1] + dev_title_length_counter[i]
    for i in range(1, 640):
        train_content_length_accumulate[i] = train_content_length_accumulate[i - 1] + train_content_length_counter[i]
        dev_content_length_accumulate[i] = dev_content_length_accumulate[i - 1] + dev_content_length_counter[i]
    title_avg_len = 0
    for title_len in title_lens.values():
        title_avg_len += title_len
    title_avg_len /= len(title_lens)
    content_avg_len = 0
    for content_len in content_lens.values():
        content_avg_len += content_len
    content_avg_len /= len(content_lens)

    print('word num :', len(word_counter))
    word_counter_list = [[word, word_counter[word]] for word in word_counter]
    word_counter_list.sort(key=lambda x: x[1], reverse=True)
    filtered_word_counter_list = list(filter(lambda x: x[1] >= 3, word_counter_list))
    print('filtered word num :', len(filtered_word_counter_list))
    print('title average length :', title_avg_len)
    print('content average length :', content_avg_len)
    print('train num :', train_cnt)
    print('train max title length :', train_max_title_len)
    print('train average title length : %.3f' % (train_title_len / train_cnt))
    print('train max content length :', train_max_content_len)
    print('train average content length : %.3f' % (train_content_len / train_cnt))
    print('dev num :', dev_cnt)
    print('dev max title length :', dev_max_title_len)
    print('dev average title length : %.3f' % (dev_title_len / dev_cnt))
    print('dev max content length :', dev_max_content_len)
    print('dev average content length : %.3f' % (dev_content_len / dev_cnt))
    print('train title length <= 8 :', train_title_length_accumulate[8] / train_cnt)
    print('train title length <= 16 :', train_title_length_accumulate[16] / train_cnt)
    print('train title length <= 24 :', train_title_length_accumulate[24] / train_cnt)
    print('train title length <= 32 :', train_title_length_accumulate[32] / train_cnt)
    print('train title length <= 48 :', train_title_length_accumulate[48] / train_cnt)
    print('train title length <= 64 :', train_title_length_accumulate[64] / train_cnt)
    print('train content length <= 16 :', train_content_length_accumulate[16] / train_cnt)
    print('train content length <= 32 :', train_content_length_accumulate[32] / train_cnt)
    print('train content length <= 48 :', train_content_length_accumulate[48] / train_cnt)
    print('train content length <= 64 :', train_content_length_accumulate[64] / train_cnt)
    print('train content length <= 96 :', train_content_length_accumulate[96] / train_cnt)
    print('train content length <= 128 :', train_content_length_accumulate[128] / train_cnt)
    print('train content length <= 256 :', train_content_length_accumulate[256] / train_cnt)
    print('train content length <= 512 :', train_content_length_accumulate[512] / train_cnt)
    print('dev title length <= 8 :', dev_title_length_accumulate[8] / dev_cnt)
    print('dev title length <= 16 :', dev_title_length_accumulate[16] / dev_cnt)
    print('dev title length <= 24 :', dev_title_length_accumulate[24] / dev_cnt)
    print('dev title length <= 32 :', dev_title_length_accumulate[32] / dev_cnt)
    print('dev title length <= 48 :', dev_title_length_accumulate[48] / dev_cnt)
    print('dev title length <= 64 :', dev_title_length_accumulate[64] / dev_cnt)
    print('dev content length <= 16 :', dev_content_length_accumulate[16] / dev_cnt)
    print('dev content length <= 32 :', dev_content_length_accumulate[32] / dev_cnt)
    print('dev content length <= 48 :', dev_content_length_accumulate[48] / dev_cnt)
    print('dev content length <= 64 :', dev_content_length_accumulate[64] / dev_cnt)
    print('dev content length <= 96 :', dev_content_length_accumulate[96] / dev_cnt)
    print('dev content length <= 128 :', dev_content_length_accumulate[128] / dev_cnt)
    print('dev content length <= 256 :', dev_content_length_accumulate[256] / dev_cnt)
    print('dev content length <= 512 :', dev_content_length_accumulate[512] / dev_cnt)
    print('\n')


def behavior_statistic(train_root, dev_root):
    train_cnt = 0
    dev_cnt = 0
    train_user_counter = collections.Counter()
    dev_user_counter = collections.Counter()
    dev_exclude_train_user_counter = collections.Counter()
    train_news_counter = collections.Counter()
    dev_news_counter = collections.Counter()
    dev_exclude_train_news_counter = collections.Counter()
    train_max_history_num = 0
    train_min_history_num = 1024
    train_history_num = 0
    dev_max_history_num = 0
    dev_min_history_num = 1024
    dev_history_num = 0
    train_history_distribution = {i: 0 for i in range(1000)}
    dev_history_distribution = {i: 0 for i in range(1000)}
    train_click_cnt = 0
    train_nonclick_cnt = 0
    dev_click_cnt = 0
    dev_nonclick_cnt = 0
    dev_max_num = 0
    with open(os.path.join(train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as train_behaviors_file:
        for line in train_behaviors_file:
            impression_ID, user_ID, time, history, impressions = line.split('\t')
            train_user_counter[user_ID] += 1
            history_num = 0 if len(history.strip()) == 0 else len(history.strip().split(' '))
            train_max_history_num = max(train_max_history_num, history_num)
            train_min_history_num = min(train_min_history_num, history_num)
            train_history_num += history_num
            train_history_distribution[history_num] += 1
            for h in history.strip().split(' '):
                train_news_counter[h] += 1
            for impression in impressions.strip().split(' '):
                if impression[-2:] == '-1':
                    train_news_counter[impression[:-2]] += 1
                    train_click_cnt += 1
                else:
                    train_nonclick_cnt += 1
            train_cnt += 1
    with open(os.path.join(dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_behaviors_file:
        for line in dev_behaviors_file:
            impression_ID, user_ID, time, history, impressions = line.split('\t')
            dev_user_counter[user_ID] += 1
            history_num = 0 if len(history.strip()) == 0 else len(history.strip().split(' '))
            dev_max_history_num = max(dev_max_history_num, history_num)
            dev_min_history_num = min(dev_min_history_num, history_num)
            dev_history_num += history_num
            dev_history_distribution[history_num] += 1
            if user_ID not in train_user_counter:
                dev_exclude_train_user_counter[user_ID] += 1
            for h in history.strip().split(' '):
                dev_news_counter[h] += 1
                if h not in train_news_counter:
                    dev_exclude_train_news_counter[h] += 1
            dev_max_num = max(len(impressions.strip().split(' ')), dev_max_num)
            for impression in impressions.strip().split(' '):
                if impression[-2:] == '-1':
                    dev_news_counter[impression[:-2]] += 1
                    if impression[:-2] not in train_news_counter:
                        dev_exclude_train_news_counter[impression[:-2]] += 1
                    dev_click_cnt += 1
                else:
                    dev_nonclick_cnt += 1
            dev_cnt += 1
    train_accumulate = [0 for _ in range(1000)]
    dev_accumulate = [0 for _ in range(1000)]
    train_accumulate[0] = train_history_distribution[0]
    dev_accumulate[0] = dev_history_distribution[0]
    for i in range(1, 1000):
        train_accumulate[i] = train_accumulate[i - 1] + train_history_distribution[i]
        dev_accumulate[i] = dev_accumulate[i - 1] + dev_history_distribution[i]
    print('train num :', train_cnt)
    print('dev num :', dev_cnt)
    print('train user :', len(train_user_counter))
    print('dev user :', len(dev_user_counter))
    print('dev exclude train user :', len(dev_exclude_train_user_counter))
    print('train news :', len(train_news_counter))
    print('dev news :', len(dev_news_counter))
    print('dev exclude train news :', len(dev_exclude_train_news_counter))
    print('train max history num :', train_max_history_num)
    print('train min history num :', train_min_history_num)
    print('train history num :', train_history_num / train_cnt)
    print('dev max history num :', dev_max_history_num)
    print('dev min history num :', dev_min_history_num)
    print('dev history num :', dev_history_num / dev_cnt)
    # print('train history distribution : ' + str(train_history_distribution))
    # print('dev history distribution : ' + str(dev_history_distribution))
    print('train history num = 0 :', train_accumulate[0] / train_cnt)
    print('train history num <= 25 :', train_accumulate[25] / train_cnt)
    print('train history num <= 50 :', train_accumulate[50] / train_cnt)
    print('train history num <= 100 :', train_accumulate[100] / train_cnt)
    print('train history num <= 200 :', train_accumulate[200] / train_cnt)
    print('train history num <= 250 :', train_accumulate[250] / train_cnt)
    print('train history num <= 500 :', train_accumulate[500] / train_cnt)
    print('dev history num = 0 :', dev_accumulate[0] / dev_cnt)
    print('dev history num <= 25 :', dev_accumulate[25] / dev_cnt)
    print('dev history num <= 50 :', dev_accumulate[50] / dev_cnt)
    print('dev history num <= 100 :', dev_accumulate[100] / dev_cnt)
    print('dev history num <= 200 :', dev_accumulate[200] / dev_cnt)
    print('dev history num <= 250 :', dev_accumulate[250] / dev_cnt)
    print('dev history num <= 500 :', dev_accumulate[500] / dev_cnt)
    print('train click num :', train_click_cnt)
    print('train non-click num :', train_nonclick_cnt)
    print('dev click num :', dev_click_cnt)
    print('dev non-click num :', dev_nonclick_cnt)
    print('dev max num :', dev_max_num)
    print('\n')


if __name__ == '__main__':
    news_statistic('../MIND/200000/train', '../MIND/200000/dev')
    behavior_statistic('../MIND/200000/train', '../MIND/200000/dev')
