import spacy


def data_preprocessing(data):
    nlp = spacy.load('pt_core_news_sm',
                     disable=['parser', 'ner', 'lemmatizer', 'textcat'])
    preprocessed_data = []
    for d in data:
        sentence = ' '.join(d[0])
        doc = nlp(sentence)
        pos_tags = [t.pos_ for t in doc]
        preprocessed_data.append((d[0], pos_tags, d[1]))
    return preprocessed_data


def extract_ner(x, y, exclude=None, is_bio=False):
    if len(x) != len(y):
        print("\nERROR: 'x' and 'y' must be the same size.")
        return None
    if type(exclude) is not list:
        exclude = []  # Guarda os tokens que serão excluídos do resultado
    extracted_ners = []  # Guarda dicionários com as entidades nomeadas filtradas/agrupadas
    for i in range(len(y)):
        ind_last_token = len(y[i]) - 1
        j = 0
        same_entity = []  # Guarda os índices dos tokens vizinhos que pertencem à mesma entidade nomeada
        ners = {}  # Guarda o índice do token (como chave) e uma lista de tokens vizinhos que pertencem à mesma entidade
        has_began = False  # Indica se um label começou com 'B-', no caso do formato BIO
        # Percorre cada um dos tipos de tokens da sentença e, se houver, verifica se o proximo tipo de token é igual
        while j <= ind_last_token:
            if y[i][j] not in exclude:
                if j not in same_entity:
                    same_entity.append(j)
                    if is_bio and len(y[i][j]) >= 2 and y[i][j][:2].upper() == 'B-':  # Se é do formato BIO e é início
                        has_began = True
                if j + 1 <= ind_last_token:
                    if has_began and len(y[i][j + 1]) >= 2 and y[i][j + 1][:2].upper() == 'I-' \
                            and y[i][j][2:] == y[i][j + 1][2:]:
                        same_entity.append(j + 1)
                    elif not is_bio and y[i][j] == y[i][j + 1]:
                        same_entity.append(j + 1)
                    else:
                        ners[j] = same_entity
                        same_entity = []
                        has_began = False
                else:
                    ners[j] = same_entity
                    has_began = False
            j += 1
        # Processa as ners extraidas da sentença desta iteração
        if ners:
            sent = x[i]
            ners_aux = []
            for ind_token, filtered_tokens in ners.items():
                ner = ''
                for t in filtered_tokens:
                    ner += sent[t] + ' '
                # Trata o caso do formato BIO e retira os 'B-' e 'I-' dos nomes dos labels
                if is_bio and len(y[i][ind_token]) >= 2:
                    label = y[i][ind_token][2:]
                else:
                    label = y[i][ind_token]
                ners_aux.append((ner.strip(), label))
            extracted_ners.append(ners_aux)
        else:
            extracted_ners.append([])  # Para manter a ordem das sentenças, caso não seja extraida nenhuma ner
    return extracted_ners
