import os
import requests
import zipfile
import json
import random
import shutil
import collections


def download_extract(url, data_file, specific_extract_path=None):
    if os.path.exists(data_file):
        return
    buffer_size = 16 * 1024
    if not os.path.exists(data_file + '.zip'):
        print('Downloading: ' + url)
        with requests.get(url, stream=True) as r, open(data_file + '.zip', 'wb') as f:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=buffer_size):
                if chunk:
                    f.write(chunk)
    if specific_extract_path is not None:
        with zipfile.ZipFile(data_file + '.zip', 'r') as f:
            f.extractall(specific_extract_path)
    else:
        with zipfile.ZipFile(data_file + '.zip', 'r') as f:
            f.extractall(data_file)

def download_extract_MIND_dataset():
    MIND_data_files = ['../MIND/train/news.tsv', '../MIND/train/behaviors.tsv', '../MIND/train/entity_embedding.vec',
                       '../MIND/dev/news.tsv', '../MIND/dev/behaviors.tsv', '../MIND/dev/entity_embedding.vec',
                       '../MIND/wikidata-graph/wikidata-graph.tsv']
    if all([os.path.exists(f) for f in MIND_data_files]):
        return
    else:
        if not os.path.exists('../MIND'):
            os.mkdir('../MIND')
        print('Downloading MIND dataset')
        # these data files are quite large, manual downloading and extracting are suggested if the `download_extract` consumes too much time
        download_extract('https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip', '../MIND/train')                                             # large version of MIND train dataset
        download_extract('https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip', '../MIND/dev')                                                 # large version of MIND dev dataset
        download_extract('https://mind201910.blob.core.windows.net/knowledge-graph/wikidata-graph.zip', '../MIND/wikidata-graph', specific_extract_path='../MIND') # knowledge graph file for the baseline DKN


def sampling_MIND_dataset(sample_num: int):
    print('Sampling MIND dataset, sampling num:', sample_num)
    # 1. randomly sample by users
    user_set = set()
    with open('../MIND/train/behaviors.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
            user_set.add(user_ID)
    with open('../MIND/dev/behaviors.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
            user_set.add(user_ID)
    user_list = list(user_set)
    random.shuffle(user_list)
    assert sample_num <= len(user_list), 'sample num must be less than or equal to 1000000'
    sample_user_list = random.sample(user_list, sample_num)
    if not os.path.exists('../MIND/%d' % sample_num):
        os.mkdir('../MIND/%d' % sample_num)
    if not os.path.exists('../MIND/%d/train' % sample_num):
        os.mkdir('../MIND/%d/train' % sample_num)
    if not os.path.exists('../MIND/%d/dev' % sample_num):
        os.mkdir('../MIND/%d/dev' % sample_num)
    if not os.path.exists('../MIND/%d/test' % sample_num):
        os.mkdir('../MIND/%d/test' % sample_num)
    with open('../MIND/%d/sample_users.json' % sample_num, 'w', encoding='utf-8') as f:
        json.dump(sample_user_list, f)
    sampled_user_set = set(sample_user_list)
    # 2. write sampled behavior file
    with open('../MIND/train/behaviors.tsv', 'r', encoding='utf-8') as f:
        with open('../MIND/%d/train/behaviors.tsv' % sample_num, 'w', encoding='utf-8') as train_f:
            for line in f:
                impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
                if user_ID in sampled_user_set:
                    train_f.write(line)
    cnt = 0
    with open('../MIND/dev/behaviors.tsv', 'r', encoding='utf-8') as f:
        with open('../MIND/%d/dev/behaviors.tsv' % sample_num, 'w', encoding='utf-8') as dev_f:
            with open('../MIND/%d/test/behaviors.tsv' % sample_num, 'w', encoding='utf-8') as test_f:
                for line in f:
                    impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
                    if user_ID in sampled_user_set:
                        if cnt % 2 == 0:
                            dev_f.write(line)  # half-split for dev
                        else:
                            test_f.write(line) # half-split for test
                        cnt += 1
    # 3. write sampled news file
    for mode in ['train', 'dev', 'test']:
        with open('../MIND/%d/%s/behaviors.tsv' % (sample_num, mode), 'r', encoding='utf-8') as f:
            news_set = set()
            for line in f:
                impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
                if len(history) > 0:
                    news = history.split(' ')
                    for n in news:
                        news_set.add(n)
                if len(impressions) > 0:
                    news = impressions.split(' ')
                    for n in news:
                        news_set.add(n[:-2])
            with open('../MIND/%s/news.tsv' % ('dev' if mode == 'test' else mode), 'r', encoding='utf-8') as _f:
                with open('../MIND/%d/%s/news.tsv' % (sample_num, mode), 'w', encoding='utf-8') as __f:
                    for line in _f:
                        news_ID, category, subCategory, title, abstract, _, _, _ = line.split('\t')
                        if news_ID in news_set:
                            __f.write(line)
    # 4. copy entity embedding file
    shutil.copyfile('../MIND/train/entity_embedding.vec', '../MIND/%d/train/entity_embedding.vec' % sample_num)
    shutil.copyfile('../MIND/dev/entity_embedding.vec', '../MIND/%d/dev/entity_embedding.vec' % sample_num)
    shutil.copyfile('../MIND/dev/entity_embedding.vec', '../MIND/%d/test/entity_embedding.vec' % sample_num)
    # 5. generate context embedding file
    entity_embeddings = {}
    entity_embedding_files = ['../MIND/train/entity_embedding.vec', '../MIND/dev/entity_embedding.vec']
    for entity_embedding_file in entity_embedding_files:
        with open(entity_embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line.strip()) > 0:
                    terms = line.strip().split('\t')
                    assert len(terms) == 101
                    entity_embeddings[terms[0]] = list(map(float, terms[1:]))
    entity_embedding_relation = collections.defaultdict(set)
    with open('../MIND/wikidata-graph/wikidata-graph.tsv', 'r', encoding='utf-8') as wikidata_graph_f:
        for line in wikidata_graph_f:
            if len(line.strip()) > 0:
                terms = line.strip().split('\t')
                entity_embedding_relation[terms[0]].add(terms[2])
                entity_embedding_relation[terms[2]].add(terms[0])
    context_embeddings = {}
    for entity in entity_embeddings:
        entity_embedding = entity_embeddings[entity]
        context_embedding = [entity_embedding[i] for i in range(100)]
        cnt = 1
        for _entity in entity_embedding_relation[entity]:
            if _entity in entity_embeddings:
                embedding = entity_embeddings[_entity]
                for i in range(100):
                    context_embedding[i] += embedding[i]
                cnt += 1
        for i in range(100):
            context_embedding[i] /= cnt
        context_embeddings[entity] = context_embedding
    for mode in ['train', 'dev', 'test']:
        with open('../MIND/%s/entity_embedding.vec' % ('dev' if mode == 'test' else mode), 'r', encoding='utf-8') as entity_embedding_f:
            with open('../MIND/%d/%s/context_embedding.vec' % (sample_num, mode), 'w', encoding='utf-8') as context_embedding_f:
                for line in entity_embedding_f:
                    if len(line.strip()) > 0:
                        entity = line.split('\t')[0]
                        context_embedding_f.write(entity + '\t' + '\t'.join(list(map(str, context_embeddings[entity]))) + '\n')


def prepare_sampled_MIND_dataset():
    required_dataset_files = [
        '../MIND/200000/train/behaviors.tsv', '../MIND/200000/train/news.tsv', '../MIND/200000/train/entity_embedding.vec', '../MIND/200000/train/context_embedding.vec',
        '../MIND/200000/dev/behaviors.tsv', '../MIND/200000/dev/news.tsv', '../MIND/200000/dev/entity_embedding.vec', '../MIND/200000/dev/context_embedding.vec',
        '../MIND/200000/test/behaviors.tsv', '../MIND/200000/test/news.tsv', '../MIND/200000/test/entity_embedding.vec', '../MIND/200000/test/context_embedding.vec'
    ]
    missing_dataset_file = False
    for required_dataset_file in required_dataset_files:
        if not os.path.exists(required_dataset_file): # if a dataset file is missing
            missing_dataset_file = True
            break
    if missing_dataset_file:
        download_extract_MIND_dataset()
        sampling_MIND_dataset(sample_num=200000)


if __name__ == '__main__':
    prepare_sampled_MIND_dataset()
