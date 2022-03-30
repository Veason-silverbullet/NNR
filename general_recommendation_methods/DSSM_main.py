import os
import shutil
import platform
from tqdm import tqdm
from DSSM_util import Config
from DSSM_model import DSSM
from DSSM_dataset import TF_IDF_Train_Dataset
from DSSM_dataset import TF_IDF_DevTest_Dataset
from DSSM_util import get_run_index
from DSSM_util import compute_scores
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, config: Config, model: nn.Module):
        self.model = model
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        self.train_dataset = TF_IDF_Train_Dataset(config)
        self.dev_dataset = TF_IDF_DevTest_Dataset(config, 'dev')
        self.run_index = get_run_index()
        if not os.path.exists('models/DSSM/#' + str(self.run_index)):
            os.mkdir('models/DSSM/#' + str(self.run_index))
        if not os.path.exists('best_model/DSSM/#' + str(self.run_index)):
            os.mkdir('best_model/DSSM/#' + str(self.run_index))
        if not os.path.exists('dev/res/DSSM/#' + str(self.run_index)):
            os.mkdir('dev/res/DSSM/#' + str(self.run_index))
        config.write_config_json('configs/DSSM/#' + str(self.run_index) + '.json')
        self.batch_size = config.batch_size
        self.epoch = config.epoch
        self.gradient_clip_norm = config.gradient_clip_norm
        self.dev_criterion = config.dev_criterion
        self.early_stopping_epoch = config.early_stopping_epoch
        self.auc = []
        self.mrr = []
        self.ndcg = []
        self.ndcg10 = []
        self.best_dev_epoch = 0
        self.best_dev_auc = 0
        self.best_dev_mrr = 0
        self.best_dev_ndcg = 0
        self.best_dev_ndcg10 = 0
        self.epoch_not_increase = 0
        print('Running : DSSM\t#' + str(self.run_index))

    def train(self):
        model = self.model
        for e in tqdm(range(self.epoch)):
            self.train_dataset.negative_sampling()
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 8 if platform.system() == 'Linux' else 0)
            model.train()
            epoch_loss = 0
            for (user_indices, user_weights, user_seq_len, news_indices, news_weights, news_seq_len) in train_dataloader:
                user_indices = user_indices.cuda(non_blocking=True)                                                # [batch_size, user_word_num]
                user_weights = user_weights.cuda(non_blocking=True)                                                # [batch_size, user_word_num]
                user_seq_len = user_seq_len.cuda(non_blocking=True)                                                # [batch_size]
                news_indices = news_indices.cuda(non_blocking=True)                                                # [batch_size, news_num, news_word_num]
                news_weights = news_weights.cuda(non_blocking=True)                                                # [batch_size, news_num, news_word_num]
                news_seq_len = news_seq_len.cuda(non_blocking=True)                                                # [batch_size, news_num]
                logits = model(user_indices, user_weights, user_seq_len, news_indices, news_weights, news_seq_len) # [batch_size, news_num]
                loss = -torch.log_softmax(logits, dim=1).select(1, 0).mean()
                epoch_loss += float(loss) * user_indices.size(0)
                self.optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
            print('Epoch %d : train done' % (e + 1))
            print('loss =', epoch_loss / len(self.train_dataset))

            # dev
            auc, mrr, ndcg, ndcg10 = compute_scores(model, self.dev_dataset, self.batch_size * 2, 'dev', 'dev/res/DSSM/#' + str(self.run_index) + '/DSSM-' + str(e + 1) + '.txt')
            self.auc.append(auc)
            self.mrr.append(mrr)
            self.ndcg.append(ndcg)
            self.ndcg10.append(ndcg10)
            print('Epoch %d : dev done\nDev criterions' % (e + 1))
            print("AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}".format(auc, mrr, ndcg, ndcg10))
            if self.dev_criterion == 'auc':
                if e == 0 or auc >= self.best_dev_auc:
                    self.best_dev_auc = auc
                    self.best_dev_epoch = e + 1
                    with open('results/DSSM/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'mrr':
                if e == 0 or mrr >= self.best_dev_mrr:
                    self.best_dev_mrr = mrr
                    self.best_dev_epoch = e + 1
                    with open('results/DSSM/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'ndcg':
                if e == 0 or ndcg >= self.best_dev_ndcg:
                    self.best_dev_ndcg = ndcg
                    self.best_dev_epoch = e + 1
                    with open('results/DSSM/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            else:
                if e == 0 or ndcg10 >= self.best_dev_ndcg10:
                    self.best_dev_ndcg10 = ndcg10
                    self.best_dev_epoch = e + 1
                    with open('results/DSSM/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1

            print('Best epoch :', self.best_dev_epoch)
            print('Best ' + self.dev_criterion + ' : ' + str(getattr(self, 'best_dev_' + self.dev_criterion)))
            torch.save({'DSSM': model.state_dict()}, 'models/DSSM/#' + str(self.run_index) + '/DSSM-' + str(e + 1))
            if self.epoch_not_increase > self.early_stopping_epoch:
                break

        shutil.copy('models/DSSM/#' + str(self.run_index) + '/DSSM-' + str(self.best_dev_epoch), 'best_model/DSSM/#' + str(self.run_index) + '/DSSM')
        print('Training : DSSM #' + str(self.run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % self.auc[self.best_dev_epoch - 1])
        print('MRR : %.4f' % self.mrr[self.best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % self.ndcg[self.best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % self.ndcg10[self.best_dev_epoch - 1])


def test(config: Config):
    model = DSSM(config)
    model.load_state_dict(torch.load(config.test_model_path, map_location=torch.device('cpu'))['DSSM'])
    model.cuda()
    test_result_path = 'test/res/DSSM/' + config.test_model_path.replace('\\', '@').replace('/', '@')
    if not os.path.exists(test_result_path):
        os.mkdir(test_result_path)
    test_dataset = TF_IDF_DevTest_Dataset(config, 'test')
    auc, mrr, ndcg, ndcg10 = compute_scores(model, test_dataset, config.batch_size * 2, 'test', test_result_path + '/DSSM.txt')
    print('Test : ' + config.test_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg, ndcg10))
    return auc, mrr, ndcg, ndcg10


if __name__ == '__main__':
    config = Config()
    config.print_log()
    if config.mode == 'train':
        model = DSSM(config)
        model.initialize()
        model.cuda()
        trainer = Trainer(config, model)
        trainer.train()
        config.test_model_path = 'best_model/DSSM/#' + str(trainer.run_index) + '/DSSM'
        result_file = 'results/DSSM/#' + str(trainer.run_index) + '-test'
        run_index = trainer.run_index
        del trainer
        del model
        auc, mrr, ndcg, ndcg10 = test(config)
        with open(result_file, 'w') as result_f:
            result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg) + '\t' + str(ndcg10) + '\n')
    else:
        pass
