import numpy as np
from src.corpora.corpora_parser import read_corpus_file, \
    generate_data, drop_duplicated_sentences


if __name__ == '__main__':

    corpus_file_dev = '../../data/corpora/le_ner/dev.conll'
    corpus_file_test = '../../data/corpora/le_ner/test.conll'
    corpus_file_train = '../../data/corpora/le_ner/train.conll'

    corpus_file_dev_clean = '../../data/corpora/le_ner/dev_clean.conll'
    corpus_file_test_clean = '../../data/corpora/le_ner/test_clean.conll'
    corpus_file_train_clean = '../../data/corpora/le_ner/train_clean.conll'

    dev = read_corpus_file(corpus_file_dev, split_char=' ')
    test = read_corpus_file(corpus_file_test, split_char=' ')
    train = read_corpus_file(corpus_file_train, split_char=' ')

    dev = np.array(dev, dtype=object)
    test = np.array(test, dtype=object)
    train = np.array(train, dtype=object)

    print('\nler_ner dev:', len(dev))
    print('ler_ner test:', len(test))
    print('ler_ner train:', len(train))

    X_dev = dev[:, 0]
    X_test = test[:, 0]
    X_train = train[:, 0]

    y_dev = dev[:, 1]
    y_test = test[:, 1]
    y_train = train[:, 1]

    print('\n---- Before cleaning ----\n')

    len_X_dev, len_X_test, len_X_train = len(X_dev), len(X_test), len(X_train)

    print('  Train set', len_X_train, '-', len(y_train))
    print('  Dev set', len_X_dev, '-', len(y_dev))
    print('  Test set', len_X_test, '-', len(y_test))

    # Cleaning duplicates in the dataset

    X_train, y_train = drop_duplicated_sentences(X_train, y_train)
    X_dev, y_dev = drop_duplicated_sentences(X_dev, y_dev)
    X_test, y_test = drop_duplicated_sentences(X_test, y_test)

    X_train, y_train = drop_duplicated_sentences(X_train, y_train,
                                                 compare_with_base=True,
                                                 base=X_test)
    X_dev, y_dev = drop_duplicated_sentences(X_dev, y_dev,
                                             compare_with_base=True,
                                             base=X_test)
    X_train, y_train = drop_duplicated_sentences(X_train, y_train,
                                                 compare_with_base=True,
                                                 base=X_dev)

    print('\n---- After cleaning ----\n')

    len_X_dev_clean, len_X_test_clean, len_X_train_clean = \
        len(X_dev), len(X_test), len(X_train)

    print('  Train set', len_X_train_clean, '-', len(y_train), '- cleaned',
          len_X_train - len_X_train_clean)
    print('  Dev set', len_X_dev_clean, '-', len(y_dev), '- cleaned',
          len_X_dev - len_X_dev_clean)
    print('  Test set', len_X_test_clean, '-', len(y_test), '- cleaned',
          len_X_test - len_X_test_clean)

    dev_data = generate_data(X_dev, y_dev, join_char=' ')
    test_data = generate_data(X_test, y_test, join_char=' ')
    train_data = generate_data(X_train, y_train, join_char=' ')

    with open(corpus_file_dev_clean, 'w', encoding='utf-8') as file:
        file.write(dev_data)

    with open(corpus_file_test_clean, 'w', encoding='utf-8') as file:
        file.write(test_data)

    with open(corpus_file_train_clean, 'w', encoding='utf-8') as file:
        file.write(train_data)
