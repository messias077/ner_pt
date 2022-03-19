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
