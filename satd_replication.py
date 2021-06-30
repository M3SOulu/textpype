import logging
import textpype.ml as ml
from textpype.ml.folds import generate_group_folds
from textpype.ml.metrics import compute_binary_metrics
from textpype import satd, nlp
from functools import partial

from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN,
                                    RandomOverSampler)

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')

df = satd.load_dataset(satd.FILENAME)
df = satd.preprocess(df)

df = df[df.keep].groupby('projectname').apply(generate_group_folds,
                                              col_name='within_fold',
                                              label='within')
df = df.reset_index(drop=True)

classifiers = {
    'lr': ml.lr,
    'balanced_lr': ml.balanced_lr,
    'rf': ml.rf,
    'balanced_rf': ml.balanced_rf,
    'balanced_sub_rf': ml.balanced_sub_rf,
    'xgb': ml.xgb,
    'balanced_xgb': ml.balanced_xgb,
}

vectorizers = {
    'tfidf': nlp.tfidf,
    'dtm': nlp.CountVectorizer,
}

samplers = {
    'none': None,
    'oversampler': partial(RandomOverSampler, random_state=42),
    'smote': partial(SMOTE, random_state=42),
    'adasyn': partial(ADASYN, random_state=42),
    'svmsmote': partial(SVMSMOTE, random_state=42),
    'borderlinesmote': partial(BorderlineSMOTE, random_state=42),
}

models = satd.Models(classifiers, vectorizers, samplers)

cross = models.eval(df, within=False,
                    save_path='output/cross', compress='gzip',
                    metrics_func=compute_binary_metrics)
cross.to_csv('output/cross_metrics.csv')

within = models.eval(df, within=True)
within = models.eval(df, within=True,
                     save_path='output/within', compress='gzip',
                     metrics_func=compute_binary_metrics)
within.to_csv('output/within_metrics.csv')
