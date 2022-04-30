import os
import signal
import shutil
import json
from config import Config
from MIND_corpus import MIND_Corpus
from MIND_dataset import MIND_Train_Dataset
from util import AvgMetric
from util import compute_scores
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(self, model: nn.Module, config: Config, mind_corpus: MIND_Corpus, run_index: int):
        self.model = model
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.max_history_num = config.max_history_num
        self.negative_sample_num = config.negative_sample_num
        self.loss = self.negative_log_softmax if config.click_predictor in ['dot_product', 'mlp', 'FIM'] else self.negative_log_sigmoid
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        self._dataset = config.dataset
        self.mind_corpus = mind_corpus
        self.train_dataset = MIND_Train_Dataset(mind_corpus)
        self.run_index = run_index
        self.model_dir = config.model_dir + '/#' + str(self.run_index)
        self.best_model_dir = config.best_model_dir + '/#' + str(self.run_index)
        self.dev_res_dir = config.dev_res_dir + '/#' + str(self.run_index)
        self.result_dir = config.result_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.best_model_dir):
            os.mkdir(self.best_model_dir)
        if not os.path.exists(self.dev_res_dir):
            os.mkdir(self.dev_res_dir)
        with open(config.config_dir + '/#' + str(self.run_index) + '.json', 'w', encoding='utf-8') as f:
            json.dump(config.attribute_dict, f)
        if self._dataset == 'large':
            self.prediction_dir = config.prediction_dir + '/#' + str(self.run_index)
            os.mkdir(self.prediction_dir)
        self.dev_criterion = config.dev_criterion
        self.early_stopping_epoch = config.early_stopping_epoch
        self.auc_results = []
        self.mrr_results = []
        self.ndcg5_results = []
        self.ndcg10_results = []
        self.best_dev_epoch = 0
        self.best_dev_auc = 0
        self.best_dev_mrr = 0
        self.best_dev_ndcg5 = 0
        self.best_dev_ndcg10 = 0
        self.best_dev_avg = AvgMetric(0, 0, 0, 0)
        self.epoch_not_increase = 0
        self.gradient_clip_norm = config.gradient_clip_norm
        self.model.cuda()
        print('Running : ' + self.model.model_name + '\t#' + str(self.run_index))

    def negative_log_softmax(self, logits):
        loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
        return loss

    def negative_log_sigmoid(self, logits):
        positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
        negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
        loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
        return loss

    def train(self):
        model = self.model
        for e in tqdm(range(1, self.epoch + 1)):
            self.train_dataset.negative_sampling()
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 16, pin_memory=True)
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
            print('Epoch %d : train done' % e)
            print('loss =', epoch_loss / len(self.train_dataset))

            # validation
            auc, mrr, ndcg5, ndcg10 = compute_scores(model, self.mind_corpus, self.batch_size * 3 // 2, 'dev', self.dev_res_dir + '/' + model.model_name + '-' + str(e) + '.txt', self._dataset)
            self.auc_results.append(auc)
            self.mrr_results.append(mrr)
            self.ndcg5_results.append(ndcg5)
            self.ndcg10_results.append(ndcg10)
            print('Epoch %d : dev done\nDev criterions' % e)
            print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}'.format(auc, mrr, ndcg5, ndcg10))
            if self.dev_criterion == 'auc':
                if auc >= self.best_dev_auc:
                    self.best_dev_auc = auc
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'mrr':
                if mrr >= self.best_dev_mrr:
                    self.best_dev_mrr = mrr
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'ndcg5':
                if ndcg5 >= self.best_dev_ndcg5:
                    self.best_dev_ndcg5 = ndcg5
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'ndcg10':
                if ndcg10 >= self.best_dev_ndcg10:
                    self.best_dev_ndcg10 = ndcg10
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            else:
                avg = AvgMetric(auc, mrr, ndcg5, ndcg10)
                if avg >= self.best_dev_avg:
                    self.best_dev_avg = avg
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1

            print('Best epoch :', self.best_dev_epoch)
            print('Best ' + self.dev_criterion + ' : ' + str(getattr(self, 'best_dev_' + self.dev_criterion)))
            torch.cuda.empty_cache()
            if self.epoch_not_increase == 0:
                torch.save({model.model_name: model.state_dict()}, self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch))
            if self.epoch_not_increase == self.early_stopping_epoch:
                break

        with open('%s/%s-%s-dev_log.txt' % (self.dev_res_dir, model.model_name, self._dataset), 'w', encoding='utf-8') as f:
            f.write('Epoch\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
            for i in range(len(self.auc_results)):
                f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (i + 1, self.auc_results[i], self.mrr_results[i], self.ndcg5_results[i], self.ndcg10_results[i]))
        shutil.copy(self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch), self.best_model_dir + '/' + model.model_name)
        print('Training : ' + model.model_name + ' #' + str(self.run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % self.auc_results[self.best_dev_epoch - 1])
        print('MRR : %.4f' % self.mrr_results[self.best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % self.ndcg5_results[self.best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % self.ndcg10_results[self.best_dev_epoch - 1])


def negative_log_softmax(logits):
    loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
    return loss

def negative_log_sigmoid(logits):
    positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
    negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
    loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
    return loss

def distributed_train(rank, model: nn.Module, config: Config, mind_corpus: MIND_Corpus, run_index: int):
    world_size = config.world_size
    model_name = model.model_name
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    config.device_id = rank
    config.set_cuda()
    model.cuda()
    loss_ = negative_log_softmax if config.click_predictor in ['dot_product', 'mlp', 'FIM'] else negative_log_sigmoid
    epoch = config.epoch
    batch_size = config.batch_size // world_size
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=config.lr, weight_decay=config.weight_decay)
    gradient_clip_norm = config.gradient_clip_norm
    train_dataset = MIND_Train_Dataset(mind_corpus)
    if rank == 0:
        model_dir = config.model_dir + '/#' + str(run_index)
        best_model_dir = config.best_model_dir + '/#' + str(run_index)
        dev_res_dir = config.dev_res_dir + '/#' + str(run_index)
        result_dir = config.result_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(best_model_dir):
            os.mkdir(best_model_dir)
        if not os.path.exists(dev_res_dir):
            os.mkdir(dev_res_dir)
        with open(config.config_dir + '/#' + str(run_index) + '.json', 'w', encoding='utf-8') as f:
            json.dump(config.attribute_dict, f)
        if config.dataset == 'large':
            prediction_dir = config.prediction_dir + '/#' + str(run_index)
            os.mkdir(prediction_dir)
        dev_criterion = config.dev_criterion
        early_stopping_epoch = config.early_stopping_epoch
        auc_results = []
        mrr_results = []
        ndcg5_results = []
        ndcg10_results = []
        best_dev_epoch = 0
        best_dev_auc = 0
        best_dev_mrr = 0
        best_dev_ndcg5 = 0
        best_dev_ndcg10 = 0
        best_dev_avg = AvgMetric(0, 0, 0, 0)
        epoch_not_increase = 0
        print('Running : ' + model_name + '\t#' + str(run_index))

    for e in tqdm(range(1, epoch + 1)):
        train_dataset.negative_sampling(rank=rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_sampler.set_epoch(e)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=batch_size // 16, pin_memory=True, sampler=train_sampler)
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

            loss = loss_(logits)
            if model.module.news_encoder.auxiliary_loss is not None:
                news_auxiliary_loss = model.module.news_encoder.auxiliary_loss.mean()
                loss += news_auxiliary_loss
            if model.module.user_encoder.auxiliary_loss is not None:
                user_encoder_auxiliary_loss = model.module.user_encoder.auxiliary_loss.mean()
                loss += user_encoder_auxiliary_loss
            epoch_loss += float(loss) * user_ID.size(0)
            optimizer.zero_grad()
            loss.backward()
            if gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        print('rank %d : Epoch %d : train done' % (rank, e))
        print('rank %d : loss = %.6f' % (rank, epoch_loss / len(train_dataset) * world_size))

        # dev
        if rank == 0:
            auc, mrr, ndcg5, ndcg10 = compute_scores(model.module, mind_corpus, batch_size * 3 // 2, 'dev', dev_res_dir + '/' + model_name + '-' + str(e) + '.txt', config.dataset)
            auc_results.append(auc)
            mrr_results.append(mrr)
            ndcg5_results.append(ndcg5)
            ndcg10_results.append(ndcg10)
            print('Epoch %d : dev done\nDev criterions' % e)
            print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}'.format(auc, mrr, ndcg5, ndcg10))
            if dev_criterion == 'auc':
                if auc >= best_dev_auc:
                    best_dev_auc = auc
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1
            elif dev_criterion == 'mrr':
                if mrr >= best_dev_mrr:
                    best_dev_mrr = mrr
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1
            elif dev_criterion == 'ndcg5':
                if ndcg5 >= best_dev_ndcg5:
                    best_dev_ndcg5 = ndcg5
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1
            elif dev_criterion == 'ndcg10':
                if ndcg10 >= best_dev_ndcg10:
                    best_dev_ndcg10 = ndcg10
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1
            else:
                avg = AvgMetric(auc, mrr, ndcg5, ndcg10)
                if avg >= best_dev_avg:
                    best_dev_avg = avg
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1

            print('Best epoch :', best_dev_epoch)
            if dev_criterion == 'auc':
                print('Best AUC : %.4f' % best_dev_auc)
            elif dev_criterion == 'mrr':
                print('Best MRR : %.4f' % best_dev_mrr)
            elif dev_criterion == 'ndcg5':
                print('Best nDCG@5 : %.4f' % best_dev_ndcg5)
            elif dev_criterion == 'ndcg10':
                print('Best nDCG@10 : %.4f' % best_dev_ndcg10)
            else:
                print('Best avg : ' + str(best_dev_avg))
            torch.cuda.empty_cache()
            if epoch_not_increase == 0:
                torch.save({model_name: model.module.state_dict()}, model_dir + '/' + model_name + '-' + str(best_dev_epoch))
            elif epoch_not_increase > early_stopping_epoch:
                break
        dist.barrier()

    if rank == 0:
        with open('%s/%s-%s-dev_log.txt' % (dev_res_dir, model_name, config.dataset), 'w', encoding='utf-8') as f:
            f.write('Epoch\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
            for i in range(len(auc_results)):
                f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (i + 1, auc_results[i], mrr_results[i], ndcg5_results[i], ndcg10_results[i]))
        print('Training : ' + model_name + ' #' + str(run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % auc_results[best_dev_epoch - 1])
        print('MRR : %.4f' % mrr_results[best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % ndcg5_results[best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % ndcg10_results[best_dev_epoch - 1])
        shutil.copy(model_dir + '/' + model_name + '-' + str(best_dev_epoch), best_model_dir + '/' + model_name)
        os.kill(os.getpid(), signal.SIGKILL)
