import os

from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, ELMoEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim import AdamW
from src.utils.utils import generate_test_file


if __name__ == '__main__':

    corpus_name = 'harem_selective'
    # corpus_name = 'harem_total'
    # corpus_name = 'paramopama'
    # corpus_name = 'le_ner'

    # Download at: http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc
    word2vec_skip_file = 'w2v_skip_s300.gensim'
    word2vec_cbow_file = 'w2v_cbow_s300.gensim'
    glove_file = 'glove_s300.gensim'

    # Download at: https://github.com/jneto04/ner-pt
    flair_forward_file = 'flairBBP_forward-pt.pt'
    flair_backward_file = 'flairBBP_backward-pt.pt'

    # Automatic download
    bert_base_embedding_path = 'neuralmind/bert-base-portuguese-cased'
    bert_large_embedding_path = 'neuralmind/bert-large-portuguese-cased'

    is_use_glove = True
    is_use_w2v_skip = False
    is_use_w2v_cbow = False

    is_use_flair = False
    is_use_elmo = False
    is_use_bert_base = False
    is_use_bert_large = False

    is_use_crf = True

    columns = None
    data_folder = None

    n_epochs = 100

    batch_size = 32

    train_file = None
    test_file = None
    val_file = None

    if corpus_name == 'le_ner':
        columns = {0: 'token', 1: 'label'}
        data_folder = 'data/corpora/le_ner'
        train_file = 'train_clean.conll'
        test_file = 'test_clean.conll'
        val_file = 'dev_clean.conll'
    elif corpus_name == 'paramopama':
        columns = {0: 'token', 1: 'label'}
        data_folder = 'data/corpora/paramopama'
        train_file = 'train.txt'
        test_file = 'test.txt'
        val_file = 'validation.txt'
    elif corpus_name == 'harem_total':
        columns = {0: 'token', 1: 'pos', 2: 'sublabel', 3: 'label'}
        data_folder = 'data/corpora/harem'
        train_file = 'train_total.txt'
        test_file = 'test_total.txt'
        val_file = 'dev_total.txt'
    elif corpus_name == 'harem_selective':
        columns = {0: 'token', 1: 'pos', 2: 'sublabel', 3: 'label'}
        data_folder = 'data/corpora/harem'
        train_file = 'train_selective.txt'
        test_file = 'test_selective.txt'
        val_file = 'dev_selective.txt'
    else:
        print('Corpus option invalid!')
        exit(0)

    if is_use_flair:
        model_dir = 'data/models/flair'
    elif is_use_bert_base:
        model_dir = 'data/models/bert_base'
    elif is_use_bert_large:
        model_dir = 'data/models/bert_large'
    elif is_use_elmo:
        model_dir = 'data/models/elmo'
    else:
        model_dir = 'data/models/bilstm'

    if is_use_w2v_skip:
        model_dir += '_w2v_skip'
    elif is_use_w2v_cbow:
        model_dir += '_w2v_cbow'
    elif is_use_glove:
        model_dir += '_glove'

    if is_use_crf:
        model_dir += '_crf'
        print('\nRunning using CRF')

    print('\n')

    model_dir = os.path.join(model_dir, corpus_name)

    os.makedirs(model_dir, exist_ok=True)

    corpus = ColumnCorpus(data_folder, columns, train_file=train_file,
                          test_file=test_file, dev_file=val_file)

    print('\nTrain len: ', len(corpus.train))
    print('Dev len: ', len(corpus.dev))
    print('Test len: ', len(corpus.test))

    print('\nTrain: ', corpus.train[0].to_tagged_string('label'))
    print('Dev: ', corpus.dev[0].to_tagged_string('label'))
    print('Test: ', corpus.test[0].to_tagged_string('label'))

    tag_type = 'label'

    # tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

    print('\nTags: ', tag_dictionary.idx2item)

    # Loading Traditional Embeddings

    traditional_embedding = None

    if is_use_w2v_skip:
        print('\nRunning using Word2vec Skip')
        traditional_embedding = WordEmbeddings(word2vec_skip_file)
    elif is_use_w2v_cbow:
        print('\nRunning using Word2vec CBOW')
        traditional_embedding = WordEmbeddings(word2vec_cbow_file)
    elif is_use_glove:
        print('\nRunning using Glove')
        traditional_embedding = WordEmbeddings(glove_file)
    else:
        print('\nNot using Traditional embedding')

    # Loading Contextual Embeddings

    embedding_types = []

    if traditional_embedding is not None:
        embedding_types.append(traditional_embedding)

    if is_use_flair:
        print('\nRunning using Flair')
        flair_embedding_forward = FlairEmbeddings(flair_forward_file)
        flair_embedding_backward = FlairEmbeddings(flair_backward_file)
        embedding_types.append(flair_embedding_forward)
        embedding_types.append(flair_embedding_backward)

    if is_use_bert_base or is_use_bert_large:
        if is_use_bert_base:
            print('\nRunning using BERT Base')
            bert_path = bert_base_embedding_path
        else:
            print('\nRunning using BERT Large')
            bert_path = bert_large_embedding_path
        bert_embedding = TransformerWordEmbeddings(
            bert_path, layers='all', allow_long_sentences=True)
        bert_embedding.max_subtokens_sequence_length = 512
        bert_embedding.stride = 0
        embedding_types.append(bert_embedding)

    if is_use_elmo:
        print('\nRunning using Elmo')
        elmo_embedding = ELMoEmbeddings('pt', embedding_mode='all')
        embedding_types.append(elmo_embedding)

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(hidden_size=256, embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=tag_type, use_crf=is_use_crf)

    trainer = ModelTrainer(tagger, corpus)

    trainer.train(model_dir,
                  optimizer=AdamW,
                  learning_rate=0.1,
                  mini_batch_size=batch_size,
                  max_epochs=n_epochs,
                  checkpoint=True,
                  num_workers=12)

    test_results_file = os.path.join(model_dir, 'test.tsv')

    new_test_file = os.path.join(model_dir, corpus_name +
                                 '_conlleval_test.tsv')

    test_results = generate_test_file(test_results_file, new_test_file)
