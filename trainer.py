import os
import shutil
import platform
import json
from config import Config
from MIND_corpus import MIND_Corpus
from MIND_dataset import MIND_Train_Dataset
from util import get_run_index
from util import compute_scores
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, config: Config, mind_corpus: MIND_Corpus):
        self.model = model
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.max_history_num = config.max_history_num
        self.negative_sample_num = config.negative_sample_num
        self.loss = self.negative_log_softmax if config.click_predictor in ['dot_product', 'mlp', 'FIM'] else self.negative_log_sigmoid
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        self.mind_corpus = mind_corpus
        self.train_dataset = MIND_Train_Dataset(mind_corpus)
        self.run_index = get_run_index(model.model_name)
        if not os.path.exists('./models/' + model.model_name + '/#' + str(self.run_index)):
            os.mkdir('./models/' + model.model_name + '/#' + str(self.run_index))
        if not os.path.exists('./best_model/' + model.model_name + '/#' + str(self.run_index)):
            os.mkdir('./best_model/' + model.model_name + '/#' + str(self.run_index))
        if not os.path.exists('./dev/res/' + model.model_name + '/#' + str(self.run_index)):
            os.mkdir('./dev/res/' + model.model_name + '/#' + str(self.run_index))
        with open('./configs/' + model.model_name + '/#' + str(self.run_index) + '.json', 'w', encoding='utf-8') as f:
            json.dump(config.attribute_dict, f)
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
        self.gradient_clip_norm = config.gradient_clip_norm
        print('Running : ' + self.model.model_name + '\t#' + str(self.run_index))

    def negative_log_softmax(self, logits):
        loss = (-torch.log_softmax(logits, dim=1)[:, 0]).mean()
        return loss

    def negative_log_sigmoid(self, logits):
        positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
        negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
        loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
        return loss

    def train(self):
        model = self.model
        for e in tqdm(range(self.epoch)):
            self.train_dataset.negative_sampling()
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 8 if platform.system() == 'Linux' else 0, pin_memory=True)
            model.train()
            epoch_loss = 0
            for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity) in train_dataloader:
                user_ID = user_ID.cuda(non_blocking=True)                                                                                                                       # [batch_size]
                user_category = user_category.cuda(non_blocking=True)                                                                                                           # [batch_size, max_history_num]
                user_subCategory = user_subCategory.cuda(non_blocking=True)                                                                                                     # [batch_size, max_history_num]
                user_title_text = user_title_text.cuda(non_blocking=True)                                                                                                       # [batch_size, max_history_num, max_title_length]
                user_title_mask = user_title_mask.cuda(non_blocking=True)                                                                                                       # [batch_size, max_history_num, max_title_length]
                user_title_entity = user_title_entity.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_title_length]
                user_content_text = user_content_text.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_content_length]
                user_content_mask = user_content_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_content_length]
                user_content_entity = user_content_entity.cuda(non_blocking=True)                                                                                               # [batch_size, max_history_num, max_content_length]
                user_history_mask = user_history_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num]
                user_history_graph = user_history_graph.cuda(non_blocking=True)                                                                                                 # [batch_size, max_history_num, max_history_num]
                user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)                                                                                 # [batch_size, category_num + 1]
                user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)                                                                           # [batch_size, max_history_num]
                news_category = news_category.cuda(non_blocking=True)                                                                                                           # [batch_size, 1 + negative_sample_num]
                news_subCategory = news_subCategory.cuda(non_blocking=True)                                                                                                     # [batch_size, 1 + negative_sample_num]
                news_title_text = news_title_text.cuda(non_blocking=True)                                                                                                       # [batch_size, 1 + negative_sample_num, max_title_length]
                news_title_mask = news_title_mask.cuda(non_blocking=True)                                                                                                       # [batch_size, 1 + negative_sample_num, max_title_length]
                news_title_entity = news_title_entity.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_title_length]
                news_content_text = news_content_text.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_content_length]
                news_content_mask = news_content_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_content_length]
                news_content_entity = news_content_entity.cuda(non_blocking=True)                                                                                               # [batch_size, 1 + negative_sample_num, max_content_length]

                logits = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                               news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity) # [batch_size, 1 + negative_sample_num]

                loss = self.loss(logits)
                if model.news_encoder.auxiliary_loss is not None:
                    news_auxiliary_loss = model.news_encoder.auxiliary_loss.mean()
                    loss += news_auxiliary_loss
                if model.user_encoder.auxiliary_loss is not None:
                    user_encoder_auxiliary_loss = model.user_encoder.auxiliary_loss.mean()
                    loss += user_encoder_auxiliary_loss
                epoch_loss += float(loss) * user_ID.size(0)
                self.optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
            print('Epoch %d : train done' % (e + 1))
            print('loss =', epoch_loss / len(self.train_dataset))

            # validation
            auc, mrr, ndcg, ndcg10 = compute_scores(model, self.mind_corpus, self.batch_size, 'dev', './dev/res/' + model.model_name + '/#' + str(self.run_index) + '/' + model.model_name + '-' + str(e + 1) + '.txt')
            self.auc.append(auc)
            self.mrr.append(mrr)
            self.ndcg.append(ndcg)
            self.ndcg10.append(ndcg10)
            print('Epoch %d : dev done\nDev criterions' % (e + 1))
            print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}'.format(auc, mrr, ndcg, ndcg10))
            if self.dev_criterion == 'auc':
                if e == 0 or auc >= self.best_dev_auc:
                    self.best_dev_auc = auc
                    self.best_dev_epoch = e + 1
                    with open('./results/' + model.model_name + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'mrr':
                if e == 0 or mrr >= self.best_dev_mrr:
                    self.best_dev_mrr = mrr
                    self.best_dev_epoch = e + 1
                    with open('./results/' + model.model_name + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'ndcg':
                if e == 0 or ndcg >= self.best_dev_ndcg:
                    self.best_dev_ndcg = ndcg
                    self.best_dev_epoch = e + 1
                    with open('./results/' + model.model_name + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            else:
                if e == 0 or ndcg10 >= self.best_dev_ndcg10:
                    self.best_dev_ndcg10 = ndcg10
                    self.best_dev_epoch = e + 1
                    with open('./results/' + model.model_name + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1

            print('Best epoch :', self.best_dev_epoch)
            print('Best ' + self.dev_criterion + ' : ' + str(getattr(self, 'best_dev_' + self.dev_criterion)))
            if self.epoch_not_increase == 0:
                torch.save({model.model_name: model.state_dict()}, './models/' + model.model_name + '/#' + str(self.run_index) + '/' + model.model_name + '-' + str(self.best_dev_epoch))
            if self.epoch_not_increase == self.early_stopping_epoch:
                break

        with open('dev/res/%s/#%d/%s-dev_log.txt' % (model.model_name, self.run_index, model.model_name), 'w', encoding='utf-8') as f:
            f.write('Epoch\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
            for i in range(len(self.auc)):
                f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (i + 1, self.auc[i], self.mrr[i], self.ndcg[i], self.ndcg10[i]))
        shutil.copy('./models/' + model.model_name + '/#' + str(self.run_index) + '/' + model.model_name + '-' + str(self.best_dev_epoch), './best_model/' + model.model_name + '/#' + str(self.run_index) + '/' + model.model_name)
        print('Training : ' + model.model_name + ' #' + str(self.run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % self.auc[self.best_dev_epoch - 1])
        print('MRR : %.4f' % self.mrr[self.best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % self.ndcg[self.best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % self.ndcg10[self.best_dev_epoch - 1])
