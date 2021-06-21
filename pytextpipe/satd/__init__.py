import re
import pandas as pd
from pytextpipe.nlp.preprocess import lemmatize, filter_stopwords, filter_size
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

FILENAME = 'data/technical_debt_dataset.csv'


def clean_text(text):
    text = text.str.lower()
    text = text.apply(lambda t: re.sub(r"(@[A-Za-z0-9]+)", "", t))
    text = text.apply(lambda t: re.sub(r"([^0-9A-Za-z \t])", "", t))
    text = text.apply(lambda t: re.sub(r"(\w+:\/\/\S+)|^rt|http.+?", "", t))
    text = text.apply(lambda t: re.sub(r"\d+", "  ", t))
    text = text.str.replace('\n', ' ').replace('\r', ' ')
    text = text.apply(lambda t: re.sub(r" +", " ", t))
    return text


def load_dataset(filename):
    df = pd.read_csv(filename, header=0, encoding='latin-1')

    df['satd'] = df.classification != 'WITHOUT_CLASSIFICATION'

    df['duplicate'] = df.duplicated(subset=['commenttext'])
    df['text_clean'] = clean_text(df.commenttext)
    df['duplicate_clean'] = df.duplicated(subset=['text_clean'])

    return df


def preprocess(df, lemmatizer=WordNetLemmatizer(),
               stopwords=stopwords.words('english')):
    df['tokens'] = df.text_clean.apply(str.split)
    df['lemmas'] = df.tokens.apply(lemmatize, lemmatizer=lemmatizer) \
                            .apply(filter_stopwords, stopwords=stopwords) \
                            .apply(filter_size) \
                            .apply(list)
    df['text_lemmatized'] = df.lemmas.apply(lambda l: ' '.join(l))
    df['keep'] = (~df.duplicate_clean) & (df.text_lemmatized.str.len() > 3)
    return df
