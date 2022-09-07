import numpy as np

from sklearn.model_selection import train_test_split
from src.corpora.corpora_parser import read_corpus_file, generate_data,\
    drop_duplicated_sentences


if __name__ == '__main__':

    corpus_file = 'data/corpora/paramopama/bio_corpus_paramopama+second_harem.txt'

    train_file = 'data/corpora/paramopama/train.txt'
    val_file = 'data/corpora/paramopama/validation.txt'
    test_file = 'data/corpora/paramopama/test.txt'

    data = read_corpus_file(corpus_file, split_char='\t')

    data = np.array(data, dtype=object)

    X = data[:, 0]
    y = data[:, 1]

    print('\nTotal before cleaning:', len(X))
    print('Data:', X[0])

    X, y = drop_duplicated_sentences(X, y)

    print('\nTotal after cleaning:', len(X))
    print('Data:', X[0])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)

    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42)

    print('\nTrain set:', len(X_train), '-', len(y_train))
    print('Validation set:', len(X_val), '-', len(y_val))
    print('Test set', len(X_test), '-', len(y_test))

    train_data = generate_data(X_train, y_train)
    validation_data = generate_data(X_val, y_val)
    test_data = generate_data(X_test, y_test)

    with open(train_file, 'w', encoding='utf-8') as file:
        file.write(train_data)

    with open(val_file, 'w', encoding='utf-8') as file:
        file.write(validation_data)

    with open(test_file, 'w', encoding='utf-8') as file:
        file.write(test_data)
