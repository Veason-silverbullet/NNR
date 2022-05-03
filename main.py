import os
import gc
import shutil
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
    run_index = get_run_index(config.result_dir)
    if config.world_size == 1:
        trainer = Trainer(model, config, mind_corpus, run_index)
        trainer.train()
        trainer = None
        del trainer
    else:
        try:
            mp.spawn(distributed_train, args=(model, config, mind_corpus, run_index), nprocs=config.world_size, join=True)
        except Exception as e:
            print(e)
            e = str(e).lower()
            if 'cuda' in e or 'pytorch' in e:
                exit()
    config.run_index = run_index
    model = None
    del model
    gc.collect()
    torch.cuda.empty_cache()


def dev(config: Config, mind_corpus: MIND_Corpus):
    model = Model(config)
    assert os.path.exists(config.dev_model_path), 'Dev model does not exist : ' + config.dev_model_path
    model.load_state_dict(torch.load(config.dev_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    dev_res_dir = os.path.join(config.dev_res_dir, config.dev_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(dev_res_dir):
        os.mkdir(dev_res_dir)
    auc, mrr, ndcg5, ndcg10 = compute_scores(model, mind_corpus, config.batch_size * 2 // config.world_size, 'dev', dev_res_dir + '/' + model.model_name + '.txt', config.dataset)
    print('Dev : ' + config.dev_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
    return auc, mrr, ndcg5, ndcg10


def test(config: Config, mind_corpus: MIND_Corpus):
    model = Model(config)
    assert os.path.exists(config.test_model_path), 'Test model does not exist : ' + config.test_model_path
    model.load_state_dict(torch.load(config.test_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    test_res_dir = os.path.join(config.test_res_dir, config.test_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(test_res_dir):
        os.mkdir(test_res_dir)
    print('test model path  : ' + config.test_model_path)
    print('test output file : ' + test_res_dir + '/' + model.model_name + '.txt')
    auc, mrr, ndcg5, ndcg10 = compute_scores(model, mind_corpus, config.batch_size * 2 // config.world_size, 'test', test_res_dir + '/' + model.model_name + '.txt', config.dataset)
    if config.dataset != 'large':
        print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
        if config.mode == 'train':
            with open(config.result_dir + '/#' + str(config.run_index) + '-test', 'w') as result_f:
                result_f.write('#' + str(config.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
        elif config.mode == 'test' and config.test_output_file != '':
            with open(config.test_output_file, 'w', encoding='utf-8') as f:
                f.write('#' + str(config.seed + 1) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
    else:
        if config.mode == 'train':
            shutil.copy(test_res_dir + '/' + model.model_name + '.txt', 'prediction/large/%s/#%d/prediction.txt' % (model.model_name, config.run_index))
            os.chdir('prediction/large/%s/#%d' % (model.model_name, config.run_index))
            os.system('zip prediction.zip prediction.txt')
            os.chdir('../../../..')


if __name__ == '__main__':
    config = Config()
    mind_corpus = MIND_Corpus(config)
    if config.mode == 'train':
        train(config, mind_corpus)
        config.test_model_path = config.best_model_dir + '/#' + str(config.run_index) + '/' + config.news_encoder + '-' + config.user_encoder
        test(config, mind_corpus)
    elif config.mode == 'dev':
        dev(config, mind_corpus)
    elif config.mode == 'test':
        test(config, mind_corpus)
