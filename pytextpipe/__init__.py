from imblearn.pipeline import Pipeline


def make_pipeline(class_counter, vectorizer, classifier, sampler=None,
                  vectorizer_args={}, classifier_args={}, sampler_args={}):
    vectorizer = vectorizer(**vectorizer_args)
    if type(vectorizer) is list:
        pipeline = vectorizer
    else:
        pipeline = [('vectorizer', vectorizer)]
    if sampler is not None:
        pipeline += [('sampler', sampler(**sampler_args))]
    pipeline += [('classifier', classifier(class_counter,
                                           **classifier_args))]
    return Pipeline(pipeline)
