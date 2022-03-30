import os
import pickle
import re
import collections
import argparse
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
parser = argparse.ArgumentParser(description='Generate tf-idf feature file')
parser.add_argument('--dataset', type=str, default='200k', choices=['200k', 'small', 'large'], help='Dataset type')
parser.add_argument('--tokenizer', type=str, default='MIND', choices=['MIND', 'NLTK'], help='Sentence tokenizer')
args = parser.parse_args()
dataset = args.dataset
train_root = '../../MIND-%s/train' % dataset
dev_root = '../../MIND-%s/dev' % dataset
test_root = '../../MIND-%s/test' % dataset
tokenizer = args.tokenizer
pat = re.compile(r"[\w]+|[.,!?;|]")
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_meta():
    stop_words = set(['.', ',', '\t', '\n', '\'', '\"', '?', '!', ';', ' ', '\n', '\t', '\r'])
    with open('NLTK_stop_words', 'r', encoding='utf-8') as stop_words_f:
        for line in stop_words_f:
            if len(line.strip()) > 0:
                stop_words.add(line.strip())
    news_ID_set = set()
    word_cnt = collections.Counter()
    for news_file in [os.path.join(train_root, 'news.tsv'), os.path.join(dev_root, 'news.tsv'), os.path.join(test_root, 'news.tsv')]:
        with open(news_file, 'r', encoding='utf-8') as news_f:
            for line in news_f:
                news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                if news_ID not in news_ID_set:
                    news_word_counter = collections.Counter()
                    words = pat.findall((title + ' ' + abstract).lower()) if tokenizer == 'MIND' else word_tokenize((title + ' ' + abstract).lower())
                    for word in words:
                        if word not in stop_words:
                            if is_number(word):
                                word = 'NUMTOKEN'
                            news_word_counter[word] += 1
                    for word in news_word_counter:
                        word_cnt[word] += 1
                    news_ID_set.add(news_ID)
    news_ID_dict = {}
    user_ID_dict = {}
    news_dict = {}
    sentence_corpus = []
    vectorizer = TfidfVectorizer()
    for i, news_file in enumerate([os.path.join(train_root, 'news.tsv'), os.path.join(dev_root, 'news.tsv'), os.path.join(test_root, 'news.tsv')]):
        with open(news_file, 'r', encoding='utf-8') as news_f:
            for line in news_f:
                news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                if news_ID not in news_dict:
                    words = pat.findall((title + ' ' + abstract).lower()) if tokenizer == 'MIND' else word_tokenize((title + ' ' + abstract).lower())
                    sentence = ''
                    for word in words:
                        if word not in stop_words and word_cnt[word] > 1:
                            if is_number(word):
                                word = 'NUMTOKEN'
                            sentence += word + ' '
                    sentence_corpus.append(sentence)
                    news_dict[news_ID] = len(news_dict)
                if news_ID not in news_ID_dict:
                    news_ID_dict[news_ID] = len(news_ID_dict)
    tfidf_matrix = vectorizer.fit_transform(sentence_corpus)
    user_history_dict = {}
    for behaviors_file in [os.path.join(train_root, 'behaviors.tsv'), os.path.join(dev_root, 'behaviors.tsv'), os.path.join(test_root, 'behaviors.tsv')]:
        with open(behaviors_file, 'r', encoding='utf-8') as behaviors_f:
            for line in behaviors_f:
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if user_ID not in user_history_dict:
                    if len(history) > 0:
                        user_history_dict[user_ID] = history.split(' ')
                    else:
                        user_history_dict[user_ID] = {}
                if user_ID not in user_ID_dict:
                    user_ID_dict[user_ID] = len(user_ID_dict)
    with open('news_ID-%s.pkl' % dataset, 'wb') as news_ID_f:
        pickle.dump(news_ID_dict, news_ID_f)
    with open('user_ID-%s.pkl' % dataset, 'wb') as user_ID_f:
        pickle.dump(user_ID_dict, user_ID_f)
    with open('offset-%s.txt' % dataset, 'w', encoding='utf-8') as f:
        f.write(str(len(news_ID_dict)) + '\n')
        f.write(str(len(user_ID_dict)) + '\n')
        f.write(str(len(vectorizer.get_feature_names())) + '\n')
    return news_dict, tfidf_matrix, user_history_dict

def generate_news_tfidf(news_dict, tfidf_matrix):
    news_tfidf = {}
    for news_ID in news_dict:
        news_matrix = tfidf_matrix[news_dict[news_ID]]
        tfidf = {}
        for word_index in news_matrix.indices:
            tfidf[word_index] = news_matrix[0, word_index]
        news_tfidf[news_ID] = tfidf
    return news_tfidf

def generate_user_tfidf(news_tfidf, user_history_dict):
    user_tfidf = {}
    for user_ID in user_history_dict:
        tfidf = {}
        for news_ID in user_history_dict[user_ID]:
            _news_tfidf = news_tfidf[news_ID]
            for word in _news_tfidf:
                if word not in tfidf:
                    tfidf[word] = _news_tfidf[word]
                else:
                    tfidf[word] = max(tfidf[word], _news_tfidf[word])
        user_tfidf[user_ID] = tfidf
    return user_tfidf


if __name__ == '__main__':
    news_dict, tfidf_matrix, user_history_dict = build_meta()
    news_tfidf = generate_news_tfidf(news_dict, tfidf_matrix)
    with open('news_tfidf-%s.pkl' % dataset, 'wb') as news_tfidf_f:
        pickle.dump(news_tfidf, news_tfidf_f)
    user_tfidf = generate_user_tfidf(news_tfidf, user_history_dict)
    with open('user_tfidf-%s.pkl' % dataset, 'wb') as user_tfidf_f:
        pickle.dump(user_tfidf, user_tfidf_f)
