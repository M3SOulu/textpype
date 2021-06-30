# textpype

Text classification pipeline

## Idea

* Dataset defined with source and preprocessing pipeline.
* Run cross validation on a dataset with a NLP pipeline.
* Dataset can define groups of data and CV can be run in a within or
  cross group setting.
* NLP pipelines, defined as individual named steps, grouped in 3
  categories: vectorizer, sampler (optional) and classifier.
