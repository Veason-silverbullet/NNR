import os
import json
import pickle
import collections
import re
from nltk.tokenize import word_tokenize
from torchtext.vocab import GloVe
from config import Config
import torch
import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

pat = re.compile(r"[\w]+|[.,!?;|]")


class MIND_Corpus:
    @staticmethod
    def preprocess(config: Config):
        user_ID_file = 'user_ID-%s.json' % config.dataset
        news_ID_file = 'news_ID-%s.json' % config.dataset
        category_file = 'category-%s.json' % config.dataset
        subCategory_file = 'subCategory-%s.json' % config.dataset
        vocabulary_file = 'vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.json'
        word_embedding_file = 'word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.pkl'
        entity_file = 'entity-%s.json' % config.dataset
        entity_embedding_file = 'entity_embedding-%s.pkl' % config.dataset
        context_embedding_file = 'context_embedding-%s.pkl' % config.dataset
        user_history_graph_file = 'user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.dataset + '.pkl'
        preprocessed_data_files = [user_ID_file, news_ID_file, category_file, subCategory_file, vocabulary_file, word_embedding_file, entity_file, entity_embedding_file, context_embedding_file, user_history_graph_file]

        if not all(list(map(os.path.exists, preprocessed_data_files))):
            user_ID_dict = {'<UNK>': 0}
            news_ID_dict = {'<PAD>': 0}
            category_dict = {}
            subCategory_dict = {}
            word_dict = {'<PAD>': 0, '<UNK>': 1}
            word_counter = collections.Counter()
            entity_dict = {'<PAD>': 0, '<UNK>': 1}
            news_category_dict = {}

            # 1. user ID dictionay
            with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
                for line in train_behaviors_f:
                    impression_ID, user_ID, time, history, impressions = line.split('\t')
                    if user_ID not in user_ID_dict:
                        user_ID_dict[user_ID] = len(user_ID_dict)
                with open(user_ID_file, 'w', encoding='utf-8') as user_ID_f:
                    json.dump(user_ID_dict, user_ID_f)

            # 2. news ID dictionay & news category dictionay & news subCategory dictionay
            for i, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                with open(os.path.join(prefix, 'news.tsv'), 'r', encoding='utf-8') as news_f:
                    for line in news_f:
                        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                        if news_ID not in news_ID_dict:
                            news_ID_dict[news_ID] = len(news_ID_dict)
                            if category not in category_dict:
                                category_dict[category] = len(category_dict)
                            if subCategory not in subCategory_dict:
                                subCategory_dict[subCategory] = len(subCategory_dict)
                            words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(title.lower())
                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0: # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter: # already appeared in training set
                                            word_counter[word] += 1
                            words = pat.findall(abstract.lower()) if config.tokenizer == 'MIND' else word_tokenize(abstract.lower())
                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0: # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter: # already appeared in training set
                                            word_counter[word] += 1
                            for entity in json.loads(title_entities):
                                WikidataId = entity['WikidataId']
                                if WikidataId not in entity_dict:
                                    entity_dict[WikidataId] = len(entity_dict)
                            for entity in json.loads(abstract_entities):
                                WikidataId = entity['WikidataId']
                                if WikidataId not in entity_dict:
                                    entity_dict[WikidataId] = len(entity_dict)
                        news_category_dict[news_ID] = category_dict[category]
            with open(news_ID_file, 'w', encoding='utf-8') as news_ID_f:
                json.dump(news_ID_dict, news_ID_f)
            with open(category_file, 'w', encoding='utf-8') as category_f:
                json.dump(category_dict, category_f)
            with open(subCategory_file, 'w', encoding='utf-8') as subCategory_f:
                json.dump(subCategory_dict, subCategory_f)

            # 3. word dictionay
            word_counter_list = [[word, word_counter[word]] for word in word_counter]
            word_counter_list.sort(key=lambda x: x[1], reverse=True) # sort by word frequency
            filtered_word_counter_list = list(filter(lambda x: x[1] >= config.word_threshold, word_counter_list))
            for i, word in enumerate(filtered_word_counter_list):
                word_dict[word[0]] = i + 2
            with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
                json.dump(word_dict, vocabulary_f)

            # 4. Glove word embedding
            if config.word_embedding_dim == 300:
                glove = GloVe(name='840B', dim=300, cache='../glove', max_vectors=10000000000)
            else:
                glove = GloVe(name='6B', dim=config.word_embedding_dim, cache='../glove', max_vectors=10000000000)
            glove_stoi = glove.stoi
            glove_vectors = glove.vectors
            glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)
            word_embedding_vectors = torch.zeros([len(word_dict), config.word_embedding_dim])
            for word in word_dict:
                index = word_dict[word]
                if index != 0:
                    if word in glove_stoi:
                        word_embedding_vectors[index, :] = glove_vectors[glove_stoi[word]]
                    else:
                        random_vector = torch.zeros(config.word_embedding_dim)
                        random_vector.normal_(mean=0, std=0.1)
                        word_embedding_vectors[index, :] = random_vector + glove_mean_vector
            with open(word_embedding_file, 'wb') as word_embedding_f:
                pickle.dump(word_embedding_vectors, word_embedding_f)

            # 5. knowledge-graph entity dictionary & eneity embedding & context embedding
            entity_embedding_vectors = torch.zeros([len(entity_dict), config.entity_embedding_dim])
            context_embedding_vectors = torch.zeros([len(entity_dict), config.context_embedding_dim])
            for prefix in [config.train_root, config.dev_root, config.test_root]:
                with open(os.path.join(prefix, 'entity_embedding.vec'), 'r', encoding='utf-8') as entity_f:
                    for line in entity_f:
                        if len(line.strip()) > 0:
                            terms = line.strip().split('\t')
                            assert len(terms) == config.entity_embedding_dim + 1, 'entity embedding dim does not match'
                            WikidataId = terms[0]
                            if WikidataId in entity_dict:
                                entity_embedding_vectors[entity_dict[WikidataId]] = torch.FloatTensor(list(map(float, terms[1:])))
            for prefix in [config.train_root, config.dev_root, config.test_root]:
                with open(os.path.join(prefix, 'context_embedding.vec'), 'r', encoding='utf-8') as context_f:
                    for line in context_f:
                        if len(line.strip()) > 0:
                            terms = line.strip().split('\t')
                            assert len(terms) == config.context_embedding_dim + 1, 'context embedding dim does not match'
                            WikidataId = terms[0]
                            if WikidataId in entity_dict:
                                context_embedding_vectors[entity_dict[WikidataId]] = torch.FloatTensor(list(map(float, terms[1:])))
            with open(entity_file, 'w', encoding='utf-8') as entity_f:
                json.dump(entity_dict, entity_f)
            with open(entity_embedding_file, 'wb') as entity_embedding_f:
                pickle.dump(entity_embedding_vectors, entity_embedding_f)
            with open(context_embedding_file, 'wb') as context_embedding_f:
                pickle.dump(context_embedding_vectors, context_embedding_f)

            # 6. user history graph
            category_num = len(category_dict)
            graph_size = config.max_history_num + category_num # graph size of |V_{n}|+|V_{p}|
            prefix_mode = ['train', 'dev', 'test']
            user_history_graph_data = {}
            for prefix_index, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                mode = prefix_mode[prefix_index]
                user_history_num = 0
                with open(os.path.join(prefix, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
                    for line in behaviors_f:
                        user_history_num += 1
                user_history_graph = np.zeros([user_history_num, graph_size, graph_size], dtype=np.float32)
                user_history_category_mask = np.zeros([user_history_num, category_num + 1], dtype=bool)
                user_history_category_indices = np.zeros([user_history_num, config.max_history_num], dtype=np.int64)
                with open(os.path.join(prefix, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
                    for line_index, line in enumerate(behaviors_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        if config.no_self_connection:
                            history_graph = np.zeros([graph_size, graph_size], dtype=np.float32)
                        else:
                            history_graph = np.identity(graph_size, dtype=np.float32)
                        history_category_mask = np.zeros(category_num + 1, dtype=bool) # extra one category index for padding news
                        history_category_indices = np.full([config.max_history_num], category_num, dtype=np.int64)
                        if len(history.strip()) > 0:
                            history_news_ID = history.split(' ')
                            offset = max(0, len(history_news_ID) - config.max_history_num)
                            history_news_num = min(len(history_news_ID), config.max_history_num)
                            for i in range(history_news_num):
                                category_index = news_category_dict[history_news_ID[i + offset]]
                                history_category_mask[category_index] = 1
                                history_category_indices[i] = category_index
                                history_graph[i, config.max_history_num + category_index] = 1 # edge of E_{p}^{1} in inter-cluster graph G2
                                history_graph[config.max_history_num + category_index, i] = 1 # edge of E_{p}^{1} in inter-cluster graph G2
                                for j in range(i + 1, history_news_num):
                                    _category_index = news_category_dict[history_news_ID[j + offset]]
                                    if category_index == _category_index:
                                        history_graph[i, j] = 1 # edge of E_{n} in intra-cluster graph G1
                                        history_graph[j, i] = 1 # edge of E_{n} in intra-cluster graph G1
                                    else:
                                        history_graph[config.max_history_num + category_index, config.max_history_num + _category_index] = 1 # edge of E_{p}^{2} in inter-cluster graph G2
                                        history_graph[config.max_history_num + _category_index, config.max_history_num + category_index] = 1 # edge of E_{p}^{2} in inter-cluster graph G2
                            if not config.no_adjacent_normalization:
                                if config.gcn_normalization_type == 'asymmetric':
                                    # Asymmetric adjacent matrix normalization: D^{-\frac{1}{2}}A
                                    D_inv = np.zeros([graph_size, graph_size], dtype=np.float32)
                                    np.fill_diagonal(D_inv, 1 / history_graph.sum(axis=1, keepdims=False))
                                    history_graph = np.matmul(D_inv, history_graph)
                                else:
                                    # Symmetric adjacent matrix normalization: D^{-\frac{1}{2}}AD^{-\frac{1}{2}}
                                    D_inv_sqrt = np.zeros([graph_size, graph_size], dtype=np.float32)
                                    np.fill_diagonal(D_inv_sqrt, np.sqrt(1 / history_graph.sum(axis=1, keepdims=False)))
                                    history_graph = np.matmul(np.matmul(D_inv_sqrt, history_graph), D_inv_sqrt)
                        user_history_graph[line_index] = history_graph
                        user_history_category_mask[line_index] = history_category_mask
                        user_history_category_indices[line_index] = history_category_indices
                    user_history_graph_data[mode + '_user_history_graph'] = user_history_graph
                    user_history_graph_data[mode + '_user_history_category_mask'] = user_history_category_mask
                    user_history_graph_data[mode + '_user_history_category_indices'] = user_history_category_indices
            with open(user_history_graph_file, 'wb') as user_history_graph_f:
                pickle.dump(user_history_graph_data, user_history_graph_f)

    def __init__(self, config: Config):
        # preprocess data
        MIND_Corpus.preprocess(config)
        with open('user_ID-%s.json' % config.dataset, 'r', encoding='utf-8') as user_ID_f:
            self.user_ID_dict = json.load(user_ID_f)
            config.user_num = len(self.user_ID_dict)
        with open('news_ID-%s.json' % config.dataset, 'r', encoding='utf-8') as news_ID_f:
            self.news_ID_dict = json.load(news_ID_f)
            self.news_num = len(self.news_ID_dict)
        with open('category-%s.json' % config.dataset, 'r', encoding='utf-8') as category_f:
            self.category_dict = json.load(category_f)
            config.category_num = len(self.category_dict)
        with open('subCategory-%s.json' % config.dataset, 'r', encoding='utf-8') as subCategory_f:
            self.subCategory_dict = json.load(subCategory_f)
            config.subCategory_num = len(self.subCategory_dict)
        with open('vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.json', 'r', encoding='utf-8') as vocabulary_f:
            self.word_dict = json.load(vocabulary_f)
            config.vocabulary_size = len(self.word_dict)
        with open('entity-%s.json' % config.dataset, 'r', encoding='utf-8') as entity_f:
            self.entity_dict = json.load(entity_f)
            config.entity_size = len(self.entity_dict)
        with open('user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.dataset + '.pkl', 'rb') as user_history_graph_f:
            user_history_data = pickle.load(user_history_graph_f)
            self.train_user_history_graph = user_history_data['train_user_history_graph']
            self.train_user_history_category_mask = user_history_data['train_user_history_category_mask']
            self.train_user_history_category_indices = user_history_data['train_user_history_category_indices']
            self.dev_user_history_graph = user_history_data['dev_user_history_graph']
            self.dev_user_history_category_mask = user_history_data['dev_user_history_category_mask']
            self.dev_user_history_category_indices = user_history_data['dev_user_history_category_indices']
            self.test_user_history_graph = user_history_data['test_user_history_graph']
            self.test_user_history_category_mask = user_history_data['test_user_history_category_mask']
            self.test_user_history_category_indices = user_history_data['test_user_history_category_indices']

        # meta data
        self.negative_sample_num = config.negative_sample_num                                           # negative sample number for training
        self.max_history_num = config.max_history_num                                                   # max history number for each training user
        self.max_title_length = config.max_title_length                                                 # max title length for each news text
        self.max_abstract_length = config.max_abstract_length                                           # max abstract length for each news text
        self.news_category = np.zeros([self.news_num], dtype=np.int32)                                  # [news_num]
        self.news_subCategory = np.zeros([self.news_num], dtype=np.int32)                               # [news_num]
        self.news_title_text = np.zeros([self.news_num, self.max_title_length], dtype=np.int32)         # [news_num, max_title_length]
        self.news_title_mask = np.zeros([self.news_num, self.max_title_length], dtype=bool)             # [news_num, max_title_length]
        self.news_title_entity = np.zeros([self.news_num, self.max_title_length], dtype=np.int32)       # [news_num, max_title_length]
        self.news_abstract_text = np.zeros([self.news_num, self.max_abstract_length], dtype=np.int32)   # [news_num, max_abstract_length]
        self.news_abstract_mask = np.zeros([self.news_num, self.max_abstract_length], dtype=bool)       # [news_num, max_abstract_length]
        self.news_abstract_entity = np.zeros([self.news_num, self.max_abstract_length], dtype=np.int32) # [news_num, max_abstract_length]
        self.train_behaviors = []                                                                       # [user_ID, [history], [history_mask], click impression, [non-click impressions], behavior_index]
        self.dev_behaviors = []                                                                         # [user_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.dev_indices = []                                                                           # index for dev
        self.test_behaviors = []                                                                        # [user_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.test_indices = []                                                                          # index for test
        self.title_word_num = 0
        self.abstract_word_num = 0

        # generate news meta data
        news_ID_set = set(['<PAD>'])
        news_lines = []
        with open(os.path.join(config.train_root, 'news.tsv'), 'r', encoding='utf-8') as train_news_f:
            for line in train_news_f:
                news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                if news_ID not in news_ID_set:
                    news_lines.append(line)
                    news_ID_set.add(news_ID)
        with open(os.path.join(config.dev_root, 'news.tsv'), 'r', encoding='utf-8') as dev_news_f:
            for line in dev_news_f:
                news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                if news_ID not in news_ID_set:
                    news_lines.append(line)
                    news_ID_set.add(news_ID)
        with open(os.path.join(config.test_root, 'news.tsv'), 'r', encoding='utf-8') as test_news_f:
            for line in test_news_f:
                news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                if news_ID not in news_ID_set:
                    news_lines.append(line)
                    news_ID_set.add(news_ID)
        assert self.news_num == len(news_ID_set), 'news num mismatch %d v.s. %d' % (self.news_num, len(news_ID_set))
        for line in news_lines:
            news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
            index = self.news_ID_dict[news_ID]
            self.news_category[index] = self.category_dict[category] if category in self.category_dict else 0
            self.news_subCategory[index] = self.subCategory_dict[subCategory] if subCategory in self.subCategory_dict else 0
            words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(title.lower())
            offsets = [-1 for _ in range(len(title))]
            offset_index = 0
            for i, word in enumerate(words):
                if i == self.max_title_length:
                    break
                if is_number(word):
                    self.news_title_text[index][i] = self.word_dict['<NUM>']
                elif word in self.word_dict:
                    self.news_title_text[index][i] = self.word_dict[word]
                else:
                    self.news_title_text[index][i] = 1
                self.news_title_mask[index][i] = 1
                while title[offset_index] in [' ', '\t']:
                    offset_index += 1
                for j in range(len(word)):
                    offsets[offset_index] = i
                    offset_index += 1
            for entity in json.loads(title_entities):
                WikidataId = entity['WikidataId']
                for offset in entity['OccurrenceOffsets']:
                    if offsets[offset] != -1 and WikidataId in self.entity_dict:
                        self.news_title_entity[index][offsets[offset]] = self.entity_dict[WikidataId]
            self.title_word_num += len(words)
            words = pat.findall(abstract.lower()) if config.tokenizer == 'MIND' else word_tokenize(abstract.lower())
            offsets = [-1 for _ in range(len(abstract))]
            offset_index = 0
            for i, word in enumerate(words):
                if i == self.max_abstract_length:
                    break
                if is_number(word):
                    self.news_abstract_text[index][i] = self.word_dict['<NUM>']
                elif word in self.word_dict:
                    self.news_abstract_text[index][i] = self.word_dict[word]
                else:
                    self.news_abstract_text[index][i] = 1
                self.news_abstract_mask[index][i] = 1
                while abstract[offset_index] in [' ', '\t']:
                    offset_index += 1
                for j in range(len(word)):
                    offsets[offset_index] = i
                    offset_index += 1
            for entity in json.loads(abstract_entities):
                WikidataId = entity['WikidataId']
                for offset in entity['OccurrenceOffsets']:
                    if offsets[offset] != -1 and WikidataId in self.entity_dict:
                        self.news_abstract_entity[index][offsets[offset]] = self.entity_dict[WikidataId]
            self.abstract_word_num += len(words)
        self.news_title_mask[0][0] = 1    # for <PAD> news
        self.news_abstract_mask[0][0] = 1 # for <PAD> news

        # generate behavior meta data
        with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
            for behavior_index, line in enumerate(train_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                click_impressions = []
                non_click_impressions = []
                for impression in impressions.strip().split(' '):
                    if impression[-2:] == '-1':
                        click_impressions.append(self.news_ID_dict[impression[:-2]])
                    else:
                        non_click_impressions.append(self.news_ID_dict[impression[:-2]])
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                    user_history_mask[:min(len(history), self.max_history_num)] = 1
                    for click_impression in click_impressions:
                        self.train_behaviors.append([self.user_ID_dict[user_ID], user_history, user_history_mask, click_impression, non_click_impressions, behavior_index])
                else:
                    for click_impression in click_impressions:
                        self.train_behaviors.append([self.user_ID_dict[user_ID], [0 for _ in range(self.max_history_num)], np.zeros([self.max_history_num], dtype=bool), click_impression, non_click_impressions, behavior_index])
        with open(os.path.join(config.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_behaviors_f:
            for dev_ID, line in enumerate(dev_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                    user_history_mask[:min(len(history), self.max_history_num)] = 1
                    for impression in impressions.strip().split(' '):
                        self.dev_indices.append(dev_ID)
                        self.dev_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history, user_history_mask, self.news_ID_dict[impression[:-2]], dev_ID])
                else:
                    for impression in impressions.strip().split(' '):
                        self.dev_indices.append(dev_ID)
                        self.dev_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, [0 for _ in range(self.max_history_num)], np.zeros([self.max_history_num], dtype=bool), self.news_ID_dict[impression[:-2]], dev_ID])
        with open(os.path.join(config.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as test_behaviors_f:
            for test_ID, line in enumerate(test_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                    user_history_mask[:min(len(history), self.max_history_num)] = 1
                    for impression in impressions.strip().split(' '):
                        self.test_indices.append(test_ID)
                        if config.dataset != 'large':
                            self.test_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history, user_history_mask, self.news_ID_dict[impression[:-2]], test_ID])
                        else:
                            self.test_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history, user_history_mask, self.news_ID_dict[impression], test_ID])
                else:
                    for impression in impressions.strip().split(' '):
                        self.test_indices.append(test_ID)
                        if config.dataset != 'large':
                            self.test_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, [0 for _ in range(self.max_history_num)], np.zeros([self.max_history_num], dtype=bool), self.news_ID_dict[impression[:-2]], test_ID])
                        else:
                            self.test_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, [0 for _ in range(self.max_history_num)], np.zeros([self.max_history_num], dtype=bool), self.news_ID_dict[impression], test_ID])
