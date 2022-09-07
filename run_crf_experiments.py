import os
import numpy as np

from src.corpora.corpora_parser import read_corpus_file
from src.nlp.nlp_parser import data_preprocessing, extract_ner
from src.machine_learning.ml_utils import convert_data
from sklearn_crfsuite import CRF
from seqeval.metrics import classification_report
from src.utils.utils import dump_report, compute_direct_match


if __name__ == '__main__':

    corpus_name = 'harem_selective'
    # corpus_name = 'harem_total'
    # corpus_name = 'le_ner'
    # corpus_name = 'paramopama'

    delimiter = '\t'

    report_dir = 'data/reports/'

    train_file = None
    test_file = None

    idx = 1

    if corpus_name == 'le_ner':
        train_file = 'data/corpora/le_ner/train_clean.conll'
        test_file = 'data/corpora/le_ner/test_clean.conll'
        delimiter = ' '
    elif corpus_name == 'paramopama':
        train_file = 'data/corpora/paramopama/train.txt'
        test_file = 'data/corpora/paramopama/test.txt'
    elif corpus_name == 'harem_total':
        train_file = 'data/corpora/harem/train_total.txt'
        test_file = 'data/corpora/harem/test_total.txt'
        delimiter = ' '
        idx = 3
    elif corpus_name == 'harem_selective':
        train_file = 'data/corpora/harem/train_selective.txt'
        test_file = 'data/corpora/harem/test_selective.txt'
        delimiter = ' '
        idx = 3
    else:
        print('Corpus option invalid!')
        exit(0)

    report_dir = os.path.join(report_dir, corpus_name)

    os.makedirs(report_dir, exist_ok=True)

    report_file = os.path.join(report_dir, corpus_name + '_crf.csv')

    train_data = read_corpus_file(train_file, split_char=delimiter,
                                  idx=idx)
    test_data = read_corpus_file(test_file, split_char=delimiter,
                                 idx=idx)

    test_data_original = np.array(test_data, dtype=object)

    print('\nCorpus:', corpus_name)

    print('\n  Train data:', len(train_data))
    print('  Test data:', len(test_data))

    print('\nPreprocessing ...')

    print('\n  Train data')

    train_data = data_preprocessing(train_data)

    print('  Test data')

    test_data = data_preprocessing(test_data)

    X_train, y_train = convert_data(train_data)
    X_test, y_test = convert_data(test_data)

    print('\nExample features:', X_train[0])
    print('Tags:', y_train[0])

    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100,
              all_possible_transitions=True)

    print('\nEvaluating CRF')

    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)

    dict_report = classification_report(y_test, y_pred, output_dict=True)

    data_conll = ''

    for data, real_tags, pred_tags in \
            zip(test_data, y_test, y_pred):
        words = data[0]
        sent = '\n'.join('{0} {1} {2}'.format(word, real_tag, pred_tag)
                         for word, real_tag, pred_tag in
                         zip(words, real_tags, pred_tags))
        sent += '\n\n'
        data_conll += sent

    print('\nReport:', dict_report)

    print('\nSaving the report in:', report_file)

    dump_report(dict_report, report_file)

    script_result_file = os.path.join(report_dir, corpus_name +
                                      '_crf.tsv')

    with open(script_result_file, 'w', encoding='utf-8') as file:
        file.write(data_conll)

    # Computing direct matching

    ner_ext_test = extract_ner(test_data_original[:, 0],
                               test_data_original[:, 1],
                               exclude=['O'], is_bio=True)

    ner_ext_pred = extract_ner(test_data_original[:, 0], y_pred,
                               exclude=['O'], is_bio=True)

    direct_match = compute_direct_match(ner_ext_test, ner_ext_pred)

    print('\nDirect match:', direct_match)

