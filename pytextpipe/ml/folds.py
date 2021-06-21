import pandas as pd
from sklearn.model_selection import StratifiedKFold


def generate_folds(df, col_name='fold', label=None, random_state=42):
    label = 'fold_{}' if label is None else label + '_fold_{}'
    folds = pd.Series([''] * len(df))
    kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for count, (_, test_index) in enumerate(kfold.split(df, df.satd)):
        folds[test_index] = label.format(count)
    df = df.reset_index(drop=True)
    df[col_name] = folds
    return df


def generate_group_folds(group, col_name='group_fold', label=None,
                         random_state=42):
    if label is None:
        label = 'group'
    label = '{}_{}'.format(label, group.name)
    return generate_folds(group, col_name, label, random_state)
