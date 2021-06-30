from textpype import make_pipeline
from textpype.ml import balanced_lr, lr
from functools import partial
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def tfidf(**kwargs):
    return [('count', CountVectorizer(**kwargs)),
            ('tfid', TfidfTransformer(use_idf=True))]


def make_tfidf_pipeline(tfidf_args={}):
    return partial(make_pipeline,
                   vectorizer=tfidf,
                   vectorizer_args=tfidf_args)
