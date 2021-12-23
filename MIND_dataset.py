from MIND_corpus import MIND_Corpus
import time
from config import Config
import torch.utils.data as data
from numpy.random import randint
from torch.utils.data import DataLoader


class MIND_Train_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus):
        self.negative_sample_num = corpus.negative_sample_num
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text =  corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_text =  corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_abstract_entity = corpus.news_abstract_entity
        self.user_history_graph = corpus.train_user_history_graph
        self.user_history_category_mask = corpus.train_user_history_category_mask
        self.user_history_category_indices = corpus.train_user_history_category_indices
        self.train_behaviors = corpus.train_behaviors
        self.train_samples = [[0 for _ in range(1 + self.negative_sample_num)] for __ in range(len(self.train_behaviors))]
        self.num = len(self.train_behaviors)

    def negative_sampling(self, rank=None):
        print('\n%sBegin negative sampling, training sample num : %d' % ('' if rank is None else ('rank ' + str(rank) + ' : '), self.num))
        start_time = time.time()
        for i, train_behavior in enumerate(self.train_behaviors):
            self.train_samples[i][0] = train_behavior[3]
            negative_samples = train_behavior[4]
            news_num = len(negative_samples)
            if news_num <= self.negative_sample_num:
                for j in range(self.negative_sample_num):
                    self.train_samples[i][j + 1] = negative_samples[j % news_num]
            else:
                used_negative_samples = set()
                for j in range(self.negative_sample_num):
                    while True:
                        k = randint(0, news_num)
                        if k not in used_negative_samples:
                            self.train_samples[i][j + 1] = negative_samples[k]
                            used_negative_samples.add(k)
                            break
        end_time = time.time()
        print('%sEnd negative sampling, used time : %.3fs' % ('' if rank is None else ('rank ' + str(rank) + ' : '), end_time - start_time))

    # user_ID                       : [1]
    # user_category                 : [max_history_num]
    # usre_subCategory              : [max_history_num]
    # user_title_text               : [max_history_num, max_title_length]
    # user_title_mask               : [max_history_num, max_title_length]
    # user_title_entity             : [max_history_num, max_title_length]
    # user_abstract_text            : [max_history_num, max_abstract_length]
    # user_abstract_mask            : [max_history_num, max_abstract_length]
    # user_abstract_entity          : [max_history_num, max_abstract_length]
    # user_history_mask             : [max_history_num]
    # user_history_graph            : [max_history_num, max_history_num]
    # user_history_category_mask    : [category_num + 1]
    # user_history_category_indices : [max_history_num]
    # news_category                 : [1 + negative_sample_num]
    # news_subCategory              : [1 + negative_sample_num]
    # news_title_text               : [1 + negative_sample_num, max_title_length]
    # news_title_mask               : [1 + negative_sample_num, max_title_length]
    # news_title_entity             : [1 + negative_sample_num, max_title_length]
    # news_abstract_text            : [1 + negative_sample_num, max_abstract_length]
    # news_abstract_mask            : [1 + negative_sample_num, max_abstract_length]
    # news_abstract_entity          : [1 + negative_sample_num, max_abstract_length]
    def __getitem__(self, index):
        train_behavior = self.train_behaviors[index]
        history_index = train_behavior[1]
        sample_index = self.train_samples[index]
        behavior_index = train_behavior[5]
        return train_behavior[0], self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], train_behavior[2], self.user_history_graph[behavior_index], self.user_history_category_mask[behavior_index], self.user_history_category_indices[behavior_index], \
               self.news_category[sample_index], self.news_subCategory[sample_index], self.news_title_text[sample_index], self.news_title_mask[sample_index], self.news_title_entity[sample_index], self.news_abstract_text[sample_index], self.news_abstract_mask[sample_index], self.news_abstract_entity[sample_index]

    def __len__(self):
        return self.num


class MIND_DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus, mode: str):
        assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text =  corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_text =  corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_abstract_entity = corpus.news_abstract_entity
        self.user_history_graph = corpus.dev_user_history_graph if mode == 'dev' else corpus.test_user_history_graph
        self.user_history_category_mask = corpus.dev_user_history_category_mask if mode == 'dev' else corpus.test_user_history_category_mask
        self.user_history_category_indices = corpus.dev_user_history_category_indices if mode == 'dev' else corpus.test_user_history_category_indices
        self.behaviors = corpus.dev_behaviors if mode == 'dev' else corpus.test_behaviors
        self.num = len(self.behaviors)

    # user_ID                        : [1]
    # user_category                  : [max_history_num]
    # user_subCategory               : [max_history_num]
    # user_title_text                : [max_history_num, max_title_length]
    # user_title_mask                : [max_history_num, max_title_length]
    # user_title_entity              : [max_history_num, max_title_length]
    # user_abstract_text             : [max_history_num, max_abstract_length]
    # user_abstract_mask             : [max_history_num, max_abstract_length]
    # user_abstract_entity           : [max_history_num, max_abstract_length]
    # user_history_mask              : [max_history_num]
    # user_history_graph             : [max_history_num, max_history_num]
    # user_history_category_mask     : [category_num + 1]
    # user_history_category_indices  : [max_history_num]
    # candidate_news_category        : [1]
    # candidate_news_subCategory     : [1]
    # candidate_news_title_text      : [max_title_length]
    # candidate_news_title_mask      : [max_title_length]
    # candidate_news_title_entity    : [max_title_lenght]
    # candidate_news_abstract_text   : [max_abstract_length]
    # candidate_news_abstract_mask   : [max_abstract_length]
    # candidate_news_abstract_entity : [max_abstract_length]
    def __getitem__(self, index):
        behavior = self.behaviors[index]
        history_index = behavior[1]
        candidate_news_index = behavior[3]
        behavior_index = behavior[4]
        return behavior[0], self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], behavior[2], self.user_history_graph[behavior_index], self.user_history_category_mask[behavior_index], self.user_history_category_indices[behavior_index], \
               self.news_category[candidate_news_index], self.news_subCategory[candidate_news_index], self.news_title_text[candidate_news_index], self.news_title_mask[candidate_news_index], self.news_title_entity[candidate_news_index], self.news_abstract_text[candidate_news_index], self.news_abstract_mask[candidate_news_index], self.news_abstract_entity[candidate_news_index]

    def __len__(self):
        return self.num


if __name__ == '__main__':
    start_time = time.time()
    config = Config()
    mind_corpus = MIND_Corpus(config)
    print('user_num :', len(mind_corpus.user_ID_dict))
    print('news_num :', len(mind_corpus.news_title_text))
    print('average title word num :', mind_corpus.title_word_num / mind_corpus.news_num)
    print('average abstract word num :', mind_corpus.abstract_word_num / mind_corpus.news_num)
    mind_train_dataset = MIND_Train_Dataset(mind_corpus)
    mind_dev_dataset = MIND_DevTest_Dataset(mind_corpus, 'dev')
    mind_test_dataset = MIND_DevTest_Dataset(mind_corpus, 'test')
    mind_train_dataset.negative_sampling()
    end_time = time.time()
    print('load time : %.3fs' % (end_time - start_time))
    print('MIND_Train_Dataset :', len(mind_train_dataset))
    train_dataloader = DataLoader(mind_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.batch_size // 16)
    for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_abstract_text, user_abstract_mask, user_abstract_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
         news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_abstract_text, news_abstract_mask, news_abstract_entity) in train_dataloader:
        print('user_ID', user_ID.size(), user_ID.dtype)
        print('user_category', user_category.size(), user_category.dtype)
        print('user_subCategory', user_subCategory.size(), user_subCategory.dtype)
        print('user_title_text', user_title_text.size(), user_title_text.dtype)
        print('user_title_mask', user_title_mask.size(), user_title_mask.dtype)
        print('user_title_entity', user_title_entity.size(), user_title_entity.dtype)
        print('user_abstract_text', user_abstract_text.size(), user_abstract_text.dtype)
        print('user_abstract_mask', user_abstract_mask.size(), user_abstract_mask.dtype)
        print('user_abstract_entity', user_abstract_entity.size(), user_abstract_entity.dtype)
        print('user_history_mask', user_history_mask.size(), user_history_mask.dtype)
        print('user_history_graph', user_history_graph.size(), user_history_graph.dtype)
        print('user_history_category_mask', user_history_category_mask.size(), user_history_category_mask.dtype)
        print('user_history_category_indices', user_history_category_indices.size(), user_history_category_indices.dtype)
        print('news_category', news_category.size(), news_category.dtype)
        print('news_subCategory', news_subCategory.size(), news_subCategory.dtype)
        print('news_title_text', news_title_text.size(), news_title_text.dtype)
        print('news_title_mask', news_title_mask.size(), news_title_mask.dtype)
        print('news_title_entity', news_title_entity.size(), news_title_entity.dtype)
        print('news_abstract_text', news_abstract_text.size(), news_abstract_text.dtype)
        print('news_abstract_mask', news_abstract_mask.size(), news_abstract_mask.dtype)
        print('news_abstract_entity', news_abstract_entity.size(), news_abstract_entity.dtype)
        break
    print('MIND_Dev_Dataset :', len(mind_dev_dataset))
    dev_dataloader = DataLoader(mind_dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.batch_size // 16)
    for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_abstract_text, user_abstract_mask, user_abstract_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
         news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_abstract_text, news_abstract_mask, news_abstract_entity) in dev_dataloader:
        print('user_ID', user_ID.size(), user_ID.dtype)
        print('user_category', user_category.size(), user_category.dtype)
        print('user_subCategory', user_subCategory.size(), user_subCategory.dtype)
        print('user_title_text', user_title_text.size(), user_title_text.dtype)
        print('user_title_mask', user_title_mask.size(), user_title_mask.dtype)
        print('user_title_entity', user_title_entity.size(), user_title_entity.dtype)
        print('user_abstract_text', user_abstract_text.size(), user_abstract_text.dtype)
        print('user_abstract_mask', user_abstract_mask.size(), user_abstract_mask.dtype)
        print('user_abstract_entity', user_abstract_entity.size(), user_abstract_entity.dtype)
        print('user_history_mask', user_history_mask.size(), user_history_mask.dtype)
        print('user_history_graph', user_history_graph.size(), user_history_graph.dtype)
        print('user_history_category_mask', user_history_category_mask.size(), user_history_category_mask.dtype)
        print('user_history_category_indices', user_history_category_indices.size(), user_history_category_indices.dtype)
        print('news_category', news_category.size(), news_category.dtype)
        print('news_subCategory', news_subCategory.size(), news_subCategory.dtype)
        print('news_title_text', news_title_text.size(), news_title_text.dtype)
        print('news_title_mask', news_title_mask.size(), news_title_mask.dtype)
        print('news_title_entity', news_title_entity.size(), news_title_entity.dtype)
        print('news_abstract_text', news_abstract_text.size(), news_abstract_text.dtype)
        print('news_abstract_mask', news_abstract_mask.size(), news_abstract_mask.dtype)
        print('news_abstract_entity', news_abstract_entity.size(), news_abstract_entity.dtype)
        break
    print(len(mind_corpus.dev_indices))
    print('MIND_Test_Dataset :', len(mind_test_dataset))
    test_dataloader = DataLoader(mind_test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.batch_size // 16)
    for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_abstract_text, user_abstract_mask, user_abstract_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
         news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_abstract_text, news_abstract_mask, news_abstract_entity) in test_dataloader:
        print('user_ID', user_ID.size(), user_ID.dtype)
        print('user_category', user_category.size(), user_category.dtype)
        print('user_subCategory', user_subCategory.size(), user_subCategory.dtype)
        print('user_title_text', user_title_text.size(), user_title_text.dtype)
        print('user_title_mask', user_title_mask.size(), user_title_mask.dtype)
        print('user_title_entity', user_title_entity.size(), user_title_entity.dtype)
        print('user_abstract_text', user_abstract_text.size(), user_abstract_text.dtype)
        print('user_abstract_mask', user_abstract_mask.size(), user_abstract_mask.dtype)
        print('user_abstract_entity', user_abstract_entity.size(), user_abstract_entity.dtype)
        print('user_history_mask', user_history_mask.size(), user_history_mask.dtype)
        print('user_history_graph', user_history_graph.size(), user_history_graph.dtype)
        print('user_history_category_mask', user_history_category_mask.size(), user_history_category_mask.dtype)
        print('user_history_category_indices', user_history_category_indices.size(), user_history_category_indices.dtype)
        print('news_category', news_category.size(), news_category.dtype)
        print('news_subCategory', news_subCategory.size(), news_subCategory.dtype)
        print('news_title_text', news_title_text.size(), news_title_text.dtype)
        print('news_title_mask', news_title_mask.size(), news_title_mask.dtype)
        print('news_title_entity', news_title_entity.size(), news_title_entity.dtype)
        print('news_abstract_text', news_abstract_text.size(), news_abstract_text.dtype)
        print('news_abstract_mask', news_abstract_mask.size(), news_abstract_mask.dtype)
        print('news_abstract_entity', news_abstract_entity.size(), news_abstract_entity.dtype)
        break
    print(len(mind_corpus.test_indices))
