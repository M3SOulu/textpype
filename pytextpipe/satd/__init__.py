import logging
import re
import pandas as pd
from pytextpipe import nlp, make_pipeline
from pytextpipe.nlp.preprocess import lemmatize, filter_stopwords, filter_size
from pytextpipe.ml.cv import fit_predict_group, fit_predict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from functools import partial

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


class Models:
    def __init__(self, classifiers, vectorizers=None, samplers=None):
        if vectorizers is None:
            vectorizers = {'dtm': nlp.CountVectorizer}
        if samplers is None:
            samplers = {'none': None}
        self.classifiers = classifiers
        self.vectorizers = vectorizers
        self.samplers = samplers

    def __repr__(self):
        s = 'SATDModels<\n vectorizer={},\n classifiers={},\n samplers={}\n>'
        return s.format(self.vectorizers, self.classifiers, self.samplers)

    def list(self):
        names = ['vectorizer', 'sampler', 'classifier']
        values = [self.vectorizers, self.samplers, self.classifiers]
        values = list(map(dict.keys, values))
        index = pd.MultiIndex.from_product(values, names=names)
        return pd.DataFrame(index=index).reset_index()

    def get_pipeline(self, model):
        classifier = model['classifier']
        vectorizer = model['vectorizer']
        sampler = model['sampler']
        return partial(make_pipeline,
                       classifier=self.classifiers[classifier],
                       vectorizer=self.vectorizers[vectorizer],
                       sampler=self.samplers[sampler])

    def eval_model(self, model, data, within=False, balance=False):
        if isinstance(model, pd.DataFrame):
            model = model.to_dict('records')[0]
        logging.info('CV for model %s', model)
        args = {'make_pipeline': self.get_pipeline(model),
                'text_col': 'text_lemmatized',
                'label_col': 'satd',
                'balance': balance}
        if within:
            args['fold_col'] = 'within_fold'
        func = fit_predict_group if within else fit_predict
        folds = func(data, 'projectname', **args)
        logging.info("Elapsed time for CV: %.2fs", sum(folds.total_time))
        return folds

    def eval(self, data, **kwargs):
        models = self.list()
        models = models.groupby(list(models.columns))
        models = models.apply(self.eval_model, data, **kwargs)
        return models
