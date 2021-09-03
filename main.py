import os
from sklearn.metrics import roc_auc_score
from config import Config
import torch
from MIND_corpus import MIND_Corpus
from model import Model
from trainer import Trainer
from util import compute_scores


def train(config: Config, mind_corpus: MIND_Corpus):
    model = Model(config)
    model.initialize()
    model.cuda()
    trainer = Trainer(model, config, mind_corpus)
    trainer.train()
    return trainer


def dev(config: Config, mind_corpus: MIND_Corpus):
    model = Model(config)
    model.load_state_dict(torch.load(config.dev_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    dev_result_path = './dev/res/' + config.dev_model_path.replace('\\', '@').replace('/', '@')
    if not os.path.exists(dev_result_path):
        os.mkdir(dev_result_path)
    auc, mrr, ndcg, ndcg10 = compute_scores(model, mind_corpus, config.batch_size, 'dev', dev_result_path + '/' + model.model_name + '.txt')
    print('Dev : ' + config.dev_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg, ndcg10))
    return auc, mrr, ndcg, ndcg10


def test(config: Config, mind_corpus: MIND_Corpus):
    model = Model(config)
    model.load_state_dict(torch.load(config.test_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    test_result_path = './test/res/' + config.test_model_path.replace('\\', '@').replace('/', '@')
    if not os.path.exists(test_result_path):
        os.mkdir(test_result_path)
    auc, mrr, ndcg, ndcg10 = compute_scores(model, mind_corpus, config.batch_size, 'test', test_result_path + '/' + model.model_name + '.txt')
    print('Test : ' + config.test_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg, ndcg10))
    if config.mode == 'test' and config.test_output_file != '':
        with open(config.test_output_file, 'w', encoding='utf-8') as f:
            f.write('#' + str(config.seed + 1) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg) + '\t' + str(ndcg10) + '\n')
    return auc, mrr, ndcg, ndcg10


if __name__ == '__main__':
    config = Config()
    mind_corpus = MIND_Corpus(config)
    if config.mode == 'train':
        trainer = train(config, mind_corpus)
        config.test_model_path = './best_model/' + trainer.model.model_name + '/#' + str(trainer.run_index) + '/' + trainer.model.model_name
        result_file = './results/' + trainer.model.model_name + '/#' + str(trainer.run_index) + '-test'
        del trainer.model
        auc, mrr, ndcg, ndcg10 = test(config, mind_corpus)
        with open(result_file, 'w') as result_f:
            result_f.write('#' + str(trainer.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg) + '\t' + str(ndcg10) + '\n')
    elif config.mode == 'dev':
        dev(config, mind_corpus)
    elif config.mode == 'test':
        test(config, mind_corpus)
