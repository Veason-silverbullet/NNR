import os
import torch
import torch.nn as nn
from MIND_corpus import MIND_Corpus
from MIND_dataset import MIND_DevTest_Dataset
from torch.utils.data import DataLoader
from evaluate import scoring


def compute_scores(model: nn.Module, mind_corpus: MIND_Corpus, batch_size: int, mode: str, result_file: str, dataset: str):
    assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
    dataloader = DataLoader(MIND_DevTest_Dataset(mind_corpus, mode), batch_size=batch_size, shuffle=False, num_workers=batch_size // 16, pin_memory=True)
    indices = (mind_corpus.dev_indices if mode == 'dev' else mind_corpus.test_indices)
    scores = torch.zeros([len(indices)]).cuda()
    index = 0
    model.eval()
    with torch.no_grad():
        for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_abstract_text, user_abstract_mask, user_abstract_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
             news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_abstract_text, news_abstract_mask, news_abstract_entity) in dataloader:
            user_ID = user_ID.cuda(non_blocking=True)
            user_category = user_category.cuda(non_blocking=True)
            user_subCategory = user_subCategory.cuda(non_blocking=True)
            user_title_text = user_title_text.cuda(non_blocking=True)
            user_title_mask = user_title_mask.cuda(non_blocking=True)
            user_title_entity = user_title_entity.cuda(non_blocking=True)
            user_abstract_text = user_abstract_text.cuda(non_blocking=True)
            user_abstract_mask = user_abstract_mask.cuda(non_blocking=True)
            user_abstract_entity = user_abstract_entity.cuda(non_blocking=True)
            user_history_mask = user_history_mask.cuda(non_blocking=True)
            user_history_graph = user_history_graph.cuda(non_blocking=True)
            user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)
            user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)
            news_category = news_category.cuda(non_blocking=True)
            news_subCategory = news_subCategory.cuda(non_blocking=True)
            news_title_text = news_title_text.cuda(non_blocking=True)
            news_title_mask = news_title_mask.cuda(non_blocking=True)
            news_title_entity = news_title_entity.cuda(non_blocking=True)
            news_abstract_text = news_abstract_text.cuda(non_blocking=True)
            news_abstract_mask = news_abstract_mask.cuda(non_blocking=True)
            news_abstract_entity = news_abstract_entity.cuda(non_blocking=True)
            batch_size = user_ID.size(0)
            news_category = news_category.unsqueeze(dim=1)
            news_subCategory = news_subCategory.unsqueeze(dim=1)
            news_title_text = news_title_text.unsqueeze(dim=1)
            news_title_mask = news_title_mask.unsqueeze(dim=1)
            news_abstract_text = news_abstract_text.unsqueeze(dim=1)
            news_abstract_mask = news_abstract_mask.unsqueeze(dim=1)
            scores[index: index+batch_size] = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_abstract_text, user_abstract_mask, user_abstract_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                                                    news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_abstract_text, news_abstract_mask, news_abstract_entity).squeeze(dim=1) # [batch_size]
            index += batch_size
    scores = scores.tolist()
    sub_scores = [[] for _ in range(indices[-1] + 1)]
    for i, index in enumerate(indices):
        sub_scores[index].append([scores[i], len(sub_scores[index])])
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    if dataset != 'large' or mode != 'test':
        with open(mode + '/ref/truth-%s.txt' % dataset, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
            auc, mrr, ndcg5, ndcg10 = scoring(truth_f, result_f)
        return auc, mrr, ndcg5, ndcg10
    else:
        return None, None, None, None


def try_to_install_torch_scatter_package():
    try:
        import torch_scatter # already installed
    except Exception as e:
        import torch
        torch_version = torch.__version__.split('+')[0]
        torch_version = torch_version[:-1] + '0' # e.g., 1.9.1 is compatible with 1.9.0
        cuda_version = None
        temp_gpu_info_file = 'gpuinfo.txt'
        os.system('nvidia-smi > ' + temp_gpu_info_file)
        with open('gpuinfo.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if 'CUDA Version:' in line:
                    cuda_info_str = line[line.find('CUDA Version:'):]
                    if '9.2' in cuda_info_str:
                        cuda_version = 'cu92'
                    elif '10.1' in cuda_info_str:
                        cuda_version = 'cu101'
                    elif '10.2' in cuda_info_str:
                        cuda_version = 'cu102'
                    elif '11.0' in cuda_info_str:
                        cuda_version = 'cu110'
                    elif '11.1' in cuda_info_str:
                        cuda_version = 'cu111'
                    break
        if os.path.exists(temp_gpu_info_file):
            os.remove(temp_gpu_info_file)
        install_flag = False
        if cuda_version is not None:
            try:
                os.system('pip install torch-scatter -f https://data.pyg.org/whl/torch-%s+%s.html' % (torch_version, cuda_version))
                install_flag = True
            except Exception as _e:
                pass
        if not install_flag:
            print('torch_scatter need to be installed by following the instruction of https://pytorch-scatter.readthedocs.io/en/latest')


def get_run_index(result_dir: str):
    assert os.path.exists(result_dir), 'result directory does not exist'
    max_index = 0
    for result_file in os.listdir(result_dir):
        if result_file.strip()[0] == '#' and result_file.strip()[-4:] == '-dev':
            index = int(result_file.strip()[1:-4])
            max_index = max(index, max_index)
    with open(result_dir + '/#' + str(max_index + 1) + '-dev', 'w', encoding='utf-8') as result_f:
        pass
    return max_index + 1
