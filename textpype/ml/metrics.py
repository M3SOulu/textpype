from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             accuracy_score, precision_score,
                             recall_score, roc_auc_score,
                             precision_recall_curve)


def compute_binary_metrics(fold):
    metrics = [('f1', f1_score),
               ('bacc', balanced_accuracy_score),
               ('acc', accuracy_score),
               ('prec', precision_score),
               ('recall', recall_score)]
    metrics = {m: f(fold['obs'], fold['preds']) for m, f in metrics}
    metrics['auc'] = roc_auc_score(fold['obs'], fold['probs'][:, 1])
    # metrics['prc'] = precision_recall_curve(fold['obs'], fold['probs'][:, 1])
    return metrics


def compute_metrics(folds, metrics_func):
    return folds.apply(metrics_func, axis=1, result_type='expand')
