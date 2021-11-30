import os
from sklearn.metrics import roc_auc_score
from config import Config
import torch
from MIND_corpus import MIND_Corpus
from model import Model
from trainer import Trainer, distributed_train
from util import compute_scores, get_run_index
import torch.multiprocessing as mp


def train(config: Config, mind_corpus: MIND_Corpus):
    model = Model(config)
    model.initialize()
    run_index = get_run_index(model.model_name)
    if config.world_size == 1:
        trainer = Trainer(model, config, mind_corpus, run_index)
        trainer.train()
    else:
        try:
            mp.spawn(distributed_train, args=(model, config, mind_corpus, run_index), nprocs=config.world_size, join=True)
        except Exception as e:
            print(e)
            e = str(e).lower()
            if 'cuda' in e or 'pytorch' in e:
                exit()
    config.run_index = run_index


def dev(config: Config, mind_corpus: MIND_Corpus):
    model = Model(config)
    model.load_state_dict(torch.load(config.dev_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    dev_result_path = './dev/res/' + config.dev_model_path.replace('\\', '@').replace('/', '@')
    if not os.path.exists(dev_result_path):
        os.mkdir(dev_result_path)
    auc, mrr, ndcg5, ndcg10 = compute_scores(model, mind_corpus, config.batch_size, 'dev', dev_result_path + '/' + model.model_name + '.txt')
    print('Dev : ' + config.dev_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
    return auc, mrr, ndcg5, ndcg10


def test(config: Config, mind_corpus: MIND_Corpus):
    model = Model(config)
    model.load_state_dict(torch.load(config.test_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    test_result_path = './test/res/' + config.test_model_path.replace('\\', '@').replace('/', '@')
    if not os.path.exists(test_result_path):
        os.mkdir(test_result_path)
    auc, mrr, ndcg5, ndcg10 = compute_scores(model, mind_corpus, config.batch_size, 'test', test_result_path + '/' + model.model_name + '.txt')
    print('Test : ' + config.test_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
    if config.mode == 'test' and config.test_output_file != '':
        with open(config.test_output_file, 'w', encoding='utf-8') as f:
            f.write('#' + str(config.seed + 1) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
    return auc, mrr, ndcg5, ndcg10


if __name__ == '__main__':
    config = Config()
    mind_corpus = MIND_Corpus(config)
    if config.mode == 'train':
        train(config, mind_corpus)
        model_name = config.news_encoder + '-' + config.user_encoder
        config.test_model_path = './best_model/' + model_name + '/#' + str(config.run_index) + '/' + model_name
        result_file = './results/' + model_name + '/#' + str(config.run_index) + '-test'
        auc, mrr, ndcg5, ndcg10 = test(config, mind_corpus)
        with open(result_file, 'w') as result_f:
            result_f.write('#' + str(config.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
    elif config.mode == 'dev':
        dev(config, mind_corpus)
    elif config.mode == 'test':
        test(config, mind_corpus)
