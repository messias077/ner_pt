"""
    Implementation based on
        https://www.aitimejournal.com/@akshay.chavan/complete-tutorial-on-named-entity-recognition-ner-using-python-and-keras
"""


def extract_sent_features(sentence):
    return [extract_features(sentence, i) for i in range(len(sentence))]


def extract_labels(sentence):
    return [label for _, _, label in sentence]


def extract_features(sentence, i):
    word = sentence[i][0]
    postag = sentence[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'word.islower()': word.islower(),
        'word[0].isupper()': word[0].isupper(),
        'word[0].islower()': word[0].islower(),
        'not word[0].isalnum()': not word[0].isalnum(),
        'not word.isalnum()': not word.isalnum(),
        'word.isalpha()': word.isalpha()
    }
    if i > 0:
        word1 = sentence[i - 1][0]
        postag1 = sentence[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:word.islower()': word1.islower()
        })
    else:
        features['BOS'] = True
    if i > 1:
        word1 = sentence[i - 2][0]
        postag1 = sentence[i - 2][1]
        features.update({
            '-2:word.lower()': word1.lower(),
            '-2:word.istitle()': word1.istitle(),
            '-2:word.isupper()': word1.isupper(),
            '-2:postag': postag1,
            '-2:postag[:2]': postag1[:2],
            '-2:word.islower()': word1.islower()
        })
    if i < len(sentence) - 1:
        word1 = sentence[i + 1][0]
        postag1 = sentence[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:word.islower()': word1.islower()
        })
    else:
        features['EOS'] = True
    if i < len(sentence) - 2:
        word1 = sentence[i + 2][0]
        postag1 = sentence[i + 2][1]
        features.update({
            '+2:word.lower()': word1.lower(),
            '+2:word.istitle()': word1.istitle(),
            '+2:word.isupper()': word1.isupper(),
            '+2:postag': postag1,
            '+2:postag[:2]': postag1[:2],
            '+2:word.islower()': word1.islower()
        })
    return features


def convert_data(data):
    sentences = []
    for d in data:
        sentences.append(list(zip(d[0], d[1], d[2])))
    x_data = [extract_sent_features(s) for s in sentences]
    y_data = [extract_labels(s) for s in sentences]
    return x_data, y_data
