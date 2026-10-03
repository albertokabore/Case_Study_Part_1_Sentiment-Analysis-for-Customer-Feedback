"""Sentiment analysis of airline customer feedback (Twitter US Airline Sentiment).

Modules
-------
config              Paths, random seed and split sizes shared by every stage.
data_loading        Loading, de-duplication, airline-label correction and splitting.
preprocessing       Tweet normalisation, tokenisation, stop-word removal, lemmatisation.
features            TF-IDF and Word2Vec feature extractors (scikit-learn compatible).
classical_models    Naive Bayes, Logistic Regression and Linear SVM pipelines + grid search.
rnn_model           Bidirectional LSTM classifier (PyTorch).
transformer_model   Fine-tuned DistilRoBERTa classifier (Hugging Face Transformers).
evaluation          Metrics, statistical tests, confusion matrices and ROC curves.
eda                 Exploratory data analysis figures and summary tables.
insights            Business-facing interpretation (key terms, error analysis, airline KPIs).
"""

__version__ = "1.0.0"
