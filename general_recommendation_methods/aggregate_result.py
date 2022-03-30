import os


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

def aggregate_criteria(model_name, criteria_list, experiment_results_f, er_f):
    sum_auc = 0
    sum_mrr = 0
    sum_ndcg5 = 0
    sum_ndcg10 = 0
    best_run_index = -1
    best_auc = 0
    best_mrr = 0
    best_ndcg5 = 0
    best_ndcg10 = 0
    for criteria in criteria_list:
        sum_auc += criteria.auc
        sum_mrr += criteria.mrr
        sum_ndcg5 += criteria.ndcg5
        sum_ndcg10 += criteria.ndcg10
        if criteria.auc > best_auc:
            best_auc = criteria.auc
            best_mrr = criteria.mrr
            best_ndcg5 = criteria.ndcg5
            best_ndcg10 = criteria.ndcg10
            best_run_index = criteria.run_index
    experiment_results_f.write('\nAvg\t%.4f\t%.4f\t%.4f\t%.4f\n' % (sum_auc / len(criteria_list), sum_mrr / len(criteria_list), sum_ndcg5 / len(criteria_list), sum_ndcg10 / len(criteria_list)))
    er_f.write('%s\t#%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (model_name, best_run_index, best_auc, best_mrr, best_ndcg5, best_ndcg10))

def aggregate_test_result():
    with open('results/experiment_results-test.tsv', 'w', encoding='utf-8') as er_f:
        for sub_dir in os.listdir('results'):
            if sub_dir in ['libfm', 'DSSM', 'wide_deep']:
                with open('results/' + sub_dir + '/experiment_results-test.tsv', 'w', encoding='utf-8') as experiment_results_f:
                    experiment_results_f.write('exp_ID\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
                    criteria_list = []
                    for result_file in os.listdir('results/' + sub_dir):
                        if result_file[0] == '#' and result_file[-5:] == '-test':
                            with open('results/' + sub_dir + '/' + result_file, 'r', encoding='utf-8') as result_f:
                                line = result_f.read()
                                if len(line.strip()) != 0:
                                    run_index, auc, mrr, ndcg5, ndcg10 = line.strip().split('\t')
                                    criteria_list.append(Criteria(int(run_index[1:]), float(auc), float(mrr), float(ndcg5), float(ndcg10)))
                    if len(criteria_list) > 0:
                        criteria_list.sort()
                        for criteria in criteria_list:
                            experiment_results_f.write(str(criteria) + '\n')
                        aggregate_criteria(sub_dir, criteria_list, experiment_results_f, er_f)


if __name__ == '__main__':
    aggregate_test_result()
