import numpy as np
import re

from unicodedata import normalize


def read_corpus_file(corpus_file, split_char='\t', idx=1):
    with open(corpus_file, encoding='utf-8') as file:
        lines = file.readlines()
    data = []
    words = []
    tags = []
    for line in lines:
        line = line.replace('\n', '')
        if line != '':
            if split_char in line:
                fragments = line.split(split_char)
                words.append(fragments[0])
                tags.append(fragments[idx])
        else:
            if len(words) > 1:
                data.append((words, tags))
            words = []
            tags = []
    return data


def drop_duplicated_sentences(x, y, compare_with_base=False, base=None):
    x_normalized = normalize_sentences(x)
    indexes_to_remove = []
    if compare_with_base:
        base_normalized = normalize_sentences(base)
        # Pesquisa comparando com um dataset base
        for s in base_normalized:
            indexes_to_remove += list(np.where(s == x_normalized)[0])
    else:
        x_normalized_set = set(x_normalized)
        # Pesquisa comparando com o próprio dataset
        for s in x_normalized_set:
            indexes_to_remove += list(np.where(s == x_normalized)[0][1:])  # Removerá a partir da segunda ocorrência
    # Remove as sentenças duplicadas
    mask = np.ones(len(x), dtype=bool)
    mask[indexes_to_remove] = False
    return x[mask], y[mask]


def generate_data(list_sentences, list_tags, join_char='\t'):
    data = ''
    for words, tags in zip(list_sentences, list_tags):
        sent = '\n'.join("{0}{1}{2}".format(w, join_char, t)
                         for w, t in zip(words, tags))
        sent += '\n\n'
        data += sent
    return data


def normalize_sentences(x):
    x_str = np.array([' '.join(s) for s in x])
    for i in range(len(x)):
        sent = x_str[i]
        # Retira os caracteres especiais
        sent = re.sub(r'[^\w\s]', '', sent)
        # Retira o espaço a mais por conta do join e retirada dos caracteres especiais
        sent = sent.replace('  ', ' ')
        sent = sent.strip()
        if sent != '':
            # Retira acentos, cedilhas, etc...
            sent = normalize('NFKD',
                             sent).encode('ASCII',
                                          'ignore').decode('ASCII')
            sent = sent.lower()
            x_str[i] = sent
    return x_str
