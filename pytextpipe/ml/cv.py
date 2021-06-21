import pandas as pd
from collections import Counter
from sklearn.utils.class_weight import compute_sample_weight


def train_test_split(fold_name, df, fold_col='fold'):
    groups = df.groupby(df[fold_col] == fold_name)
    print(groups)
    return groups.get_group(False), groups.get_group(True)


def fit(make_pipeline, text, labels, balance=False):
    pipeline = make_pipeline(class_counter=Counter(labels))
    sample_weight = None
    if balance:
        print('Computating balanced weights')
        sample_weight = compute_sample_weight(class_weight='balanced',
                                              y=labels)
    pipeline.fit(text, labels, classifier__sample_weight=sample_weight)
    return pipeline


def predict(fold_name, pipeline, text, labels):
    preds = pipeline.predict(text)
    probs = pipeline.predict_proba(text)
    return {
        'fold': fold_name,
        'pipeline': pipeline,
        'text': text,
        'obs': labels,
        'preds': preds,
        'probs': probs,
    }


def fit_predict_fold(fold_name, df, make_pipeline,
                     fold_col='fold_col',
                     label_col='label',
                     text_col='text',
                     balance=False):
    train, test = train_test_split(fold_name, df, fold_col)
    print('Training model for fold {}'.format(fold_name))
    pipeline = fit(make_pipeline, train[text_col], train[label_col], balance)
    print('Evaluating model for fold {}'.format(fold_name))
    return predict(fold_name, pipeline, test[text_col], test[label_col])


def fit_predict(df, fold_col='fold', **kwargs):
    folds = df[fold_col].unique()
    folds.sort()
    kwargs['fold_col'] = fold_col
    result = map(lambda f: fit_predict_fold(f, df, **kwargs), folds)
    return pd.DataFrame.from_records(result).set_index('fold')


def fit_predict_group(df, groups, **kwargs):
    return df.groupby(groups).apply(fit_predict, **kwargs)
