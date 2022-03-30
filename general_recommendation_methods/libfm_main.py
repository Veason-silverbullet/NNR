import os
import argparse
from generate_libfm_data import generate_libfm_data
from evaluate import scoring


parser = argparse.ArgumentParser(description='LibFM')
parser.add_argument('--dataset', type=str, default='200k', choices=['200k', 'small', 'large'], help='Dataset type')
args = parser.parse_args()
dataset = args.dataset


if not os.path.exists('news_tfidf-%s.pkl' % dataset) or not os.path.exists('user_tfidf-%s.pkl' % dataset):
    os.system('python generate_tf_idf_feature_file.py')
if not os.path.exists('train-%s.libfm' % dataset) or not os.path.exists('dev-%s.libfm' % dataset) or not os.path.exists('test-%s.libfm' % dataset):
    generate_libfm_data()
if not os.path.exists('dev'):
    os.mkdir('dev')
if not os.path.exists('dev/ref'):
    os.mkdir('dev/ref')
if not os.path.exists('dev/res'):
    os.mkdir('dev/res')
if not os.path.exists('dev/res/libfm'):
    os.mkdir('dev/res/libfm')
if not os.path.exists('test'):
    os.mkdir('test')
if not os.path.exists('test/ref'):
    os.mkdir('test/ref')
if not os.path.exists('test/res'):
    os.mkdir('test/res')
if not os.path.exists('test/res/libfm'):
    os.mkdir('test/res/libfm')
if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists('results/libfm'):
    os.mkdir('results/libfm')
if not os.path.exists('test/ref/truth.txt'):
    with open('../../MIND-%s/test/behaviors.tsv' % dataset, 'r', encoding='utf-8') as test_f:
        with open('test/ref/truth.txt', 'w', encoding='utf-8') as truth_f:
            for test_ID, line in enumerate(test_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                truth_f.write(('' if test_ID == 0 else '\n') + str(test_ID + 1) + ' ' + str(labels).replace(' ', ''))


def get_run_index():
    max_index = 0
    for result_file in os.listdir('results/libfm'):
        if result_file.strip()[0] == '#' and result_file.strip()[-5:] == '-test':
            index = int(result_file.strip()[1:-5])
            max_index = max(index, max_index)
    with open('results/libfm/#' + str(max_index + 1) + '-test', 'w', encoding='utf-8') as result_f:
        pass
    return max_index + 1

def write_result_file(probs, libfm_result_file):
    k = 0
    with open('../../MIND-%s/test/behaviors.tsv' % dataset, 'r', encoding='utf-8') as behaviors_f:
        with open(libfm_result_file, 'w', encoding='utf-8') as f:
            for i, line in enumerate(behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                num = len(impressions.strip().split(' '))
                scores = []
                for j in range(num):
                    scores.append([probs[k], j])
                    k += 1
                scores.sort(key=lambda x: x[0], reverse=True)
                result = [0 for _ in range(num)]
                for j in range(num):
                    result[scores[j][1]] = j + 1
                f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    assert len(probs) == k, str(len(probs)) + ' - ' + str(k)


if __name__ == '__main__':
    run_index = get_run_index()
    os.mkdir('test/res/libfm/%d' % run_index)
    print('Running : libfm\t#' + str(run_index))
    os.system('./libfm/bin/libFM -task r -train train-%s.libfm -test test-%s.libfm -out ./test/res/libfm/%d/libfm' % (dataset, dataset, run_index))
    probs = []
    with open('test/res/libfm/%d/libfm' % run_index, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 0:
                probs.append(float(line.strip()))
    write_result_file(probs, 'test/res/libfm/%d/libfm.txt' % run_index)
    with open('test/ref/truth.txt', 'r', encoding='utf-8') as truth_f, open('test/res/libfm/%d/libfm.txt' % run_index, 'r', encoding='utf-8') as res_f:
        auc, mrr, ndcg, ndcg10 = scoring(truth_f, res_f)
        print('AUC =', auc)
        print('MRR =', mrr)
        print('nDCG@5 =', ndcg)
        print('nDCG@10 =', ndcg10)
        with open('results/libfm/#%d-test' % run_index, 'w', encoding='utf-8') as f:
            f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg) + '\t' + str(ndcg10) + '\n')
