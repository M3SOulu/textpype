# from nltk.tokenize import word_tokenize


def lemmatize(tokens, lemmatizer):
    return map(lemmatizer.lemmatize, tokens)


def filter_stopwords(tokens, stopwords):
    return (t for t in tokens if t not in stopwords)


def filter_size(tokens, min_size=0, max_size=15):
    return (t for t in tokens if min_size < len(t) < max_size)
