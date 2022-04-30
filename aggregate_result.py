import os


model_dict = {
    'DAE-GRU': 'EBNR',
    'KCNN-CATT': 'DKN',
    'PNE-PUE': 'NPA',
    'CNN-LSTUR': 'LSTUR',
    'NAML-ATT': 'NAML',
    'MHSA-MHSA': 'NRMS',
    'HDC-FIM': 'FIM',
    'CNE-SUE': 'CNE-SUE'
}


class Criteria:
    def __init__(self, run_index, auc, mrr, ndcg5, ndcg10):
        self.run_index = run_index
        self.auc = auc
        self.mrr = mrr
        self.ndcg5 = ndcg5
        self.ndcg10 = ndcg10

    def __gt__(self, value):
        return self.run_index > value.run_index

    def __ge__(self, value):
        return self.run_index >= value.run_index

    def __lt__(self, value):
        return self.run_index < value.run_index

    def __le__(self, value):
        return self.run_index <= value.run_index

    def __str__(self):
        return '#%d\t%.4f\t%.4f\t%.4f\t%.4f' % (self.run_index, self.auc, self.mrr, self.ndcg5, self.ndcg10)

def aggregate_criteria(model_name, criteria_list, experiment_results_f):
    sum_auc = 0
    sum_mrr = 0
    sum_ndcg5 = 0
    sum_ndcg10 = 0
    for criteria in criteria_list:
        sum_auc += criteria.auc
        sum_mrr += criteria.mrr
        sum_ndcg5 += criteria.ndcg5
        sum_ndcg10 += criteria.ndcg10
    mean_auc = sum_auc / len(criteria_list)
    mean_mrr = sum_mrr / len(criteria_list)
    mean_ndcg5 = sum_ndcg5 / len(criteria_list)
    mean_ndcg10 = sum_ndcg10 / len(criteria_list)
    experiment_results_f.write('\nAvg\t%.4f\t%.4f\t%.4f\t%.4f\n' % (mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10))
    return mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10

def list_model_name():
    model_names = []
    for news_encoder in ['CNE', 'CNN', 'MHSA', 'KCNN', 'HDC', 'NAML', 'PNE', 'DAE', 'Inception', 'NAML_Title', 'NAML_Content', 'CNE_Title', 'CNE_Content', 'CNE_wo_CS', 'CNE_wo_CA']:
        for user_encoder in ['SUE', 'LSTUR', 'MHSA', 'ATT', 'CATT', 'FIM', 'PUE', 'GRU', 'OMAP', 'SUE_wo_GCN', 'SUE_wo_HCA']:
            model_names.append(news_encoder + '-' + user_encoder)
    return model_names

def aggregate_dev_result():
    for dataset in ['small', '200k', 'large']:
        if os.path.exists('results/' + dataset):
            for sub_dir in os.listdir('results/' + dataset):
                if sub_dir in list_model_name():
                    with open('results/' + dataset + '/' + sub_dir + '/experiment_results-dev.tsv', 'w', encoding='utf-8') as experiment_results_f:
                        experiment_results_f.write('exp_ID\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
                        criteria_list = []
                        for result_file in os.listdir('results/' + dataset + '/' + sub_dir):
                            if result_file[0] == '#' and result_file[-4:] == '-dev':
                                with open('results/' + dataset + '/' + sub_dir + '/' + result_file, 'r', encoding='utf-8') as result_f:
                                    line = result_f.read()
                                    if len(line.strip()) != 0:
                                        run_index, auc, mrr, ndcg5, ndcg10 = line.strip().split('\t')
                                        criteria_list.append(Criteria(int(run_index[1:]), float(auc), float(mrr), float(ndcg5), float(ndcg10)))
                        if len(criteria_list) > 0:
                            criteria_list.sort()
                            for criteria in criteria_list:
                                experiment_results_f.write(str(criteria) + '\n')
                            mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10 = aggregate_criteria(sub_dir, criteria_list, experiment_results_f)

def aggregate_test_result():
    for dataset in ['small', '200k', 'large']:
        if os.path.exists('results/' + dataset):
            with open('results/%s/overall.tsv' % dataset, 'w', encoding='utf-8') as overall_f:
                for sub_dir in os.listdir('results/' + dataset):
                    if sub_dir in list_model_name():
                        with open('results/' + dataset + '/' + sub_dir + '/experiment_results-test.tsv', 'w', encoding='utf-8') as experiment_results_f:
                            experiment_results_f.write('exp_ID\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
                            criteria_list = []
                            for result_file in os.listdir('results/' + dataset + '/' + sub_dir):
                                if result_file[0] == '#' and result_file[-5:] == '-test':
                                    with open('results/' + dataset + '/' + sub_dir + '/' + result_file, 'r', encoding='utf-8') as result_f:
                                        line = result_f.read()
                                        if len(line.strip()) != 0:
                                            run_index, auc, mrr, ndcg5, ndcg10 = line.strip().split('\t')
                                            criteria_list.append(Criteria(int(run_index[1:]), float(auc), float(mrr), float(ndcg5), float(ndcg10)))
                            if len(criteria_list) > 0:
                                criteria_list.sort()
                                for criteria in criteria_list:
                                    experiment_results_f.write(str(criteria) + '\n')
                                mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10 = aggregate_criteria(sub_dir, criteria_list, experiment_results_f)
                                overall_f.write('%s\t%.4f\t%.4f\t%.4f\t%.4f\n' % (model_dict[sub_dir] if sub_dir in model_dict else sub_dir, mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10))


if __name__ == '__main__':
    aggregate_dev_result()
    aggregate_test_result()
