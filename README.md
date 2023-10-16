Conducted sentiment analysis on a dataset of IMDb movie reviews using various machine learning classifiers and different text vectorization techniques.

Here's a summary of the key steps:

Data Loading: The IMDb movie review dataset is loaded into a Pandas DataFrame.

Data Preprocessing: Sentiment labels ('positive' and 'negative') are converted to numerical values (1 and 0).

Data Splitting: The dataset is split into training and test sets.

TF-IDF Vectorization: Text data is transformed into TF-IDF vectors for both training and test sets.

Machine Learning Models:

Decision Tree, Random Forest, LinearSVC, and Multinomial Naive Bayes classifiers are trained and evaluated.
Metrics such as F1 score, accuracy, precision, recall, ROC curve, and AUC are calculated for each classifier.
Bag of Words (BoW) Vectorization and Model Evaluation: Similar to TF-IDF, BoW vectorization is applied, and the same classifiers are trained and evaluated.

Doc2Vec Vectorization and Model Evaluation: Doc2Vec vectorization is applied using Gensim. Two classifiers (SVC and Multinomial Naive Bayes) are trained and evaluated on Doc2Vec vectors.

The code provides a comprehensive comparison of classifier performance using different text representation techniques, aiding in selecting the most suitable approach for sentiment analysis of movie reviews.





