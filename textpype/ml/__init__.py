from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def balanced_lr(class_counter, **kwargs):
    return LogisticRegression(solver='liblinear',
                              class_weight='balanced',
                              **kwargs)


# lr = partial(LogisticRegression, solver='liblinear')
# balanced_lr = partial(lr, class_weight='balanced')


def lr(class_counter, **kwargs):
    return LogisticRegression(solver='liblinear',
                              **kwargs)


def xgb(class_counter, **kwargs):
    return XGBClassifier(**kwargs)


def balanced_xgb(class_counter, **kwargs):
    scale_pos_weight = class_counter[False] / class_counter[True]
    return XGBClassifier(scale_pos_weight=scale_pos_weight, **kwargs)


def rf(class_counter, **kwargs):
    return RandomForestClassifier(**kwargs)


def balanced_rf(class_counter, **kwargs):
    return RandomForestClassifier(class_weight='balanced', **kwargs)


def balanced_sub_rf(class_counter, **kwargs):
    return RandomForestClassifier(class_weight='balanced_subsample', **kwargs)
