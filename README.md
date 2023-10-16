performing sentiment analysis on a dataset of movie reviews from IMDb. It follows these main steps:

Data Loading: The code starts by loading a CSV file containing the movie reviews and their corresponding sentiment labels (positive or negative). It uses the Pandas library to read the CSV file into a DataFrame (df).

Data Preprocessing:

The sentiment labels in the 'sentiment' column of the DataFrame are converted from 'positive' and 'negative' to numerical values, 1 and 0, respectively.
Data Splitting: The dataset is split into training and test sets using the train_test_split function from scikit-learn (X_train, X_test, y_train, y_test). The test size is set to 20% of the data, and a random seed (random_state) is provided for reproducibility.

Feature Extraction using TF-IDF Vectorization:

TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is applied to convert the text data (movie reviews) into numerical vectors.
A TfidfVectorizer is created and fitted to the training data (vectorizer.fit(X_train)).
The training and test data are transformed into TF-IDF vectors (X_train_vectorized and X_test_vectorized) using the fitted vectorizer.
Machine Learning Model Training and Evaluation:

Four different machine learning classifiers are trained and evaluated on both TF-IDF and Bag of Words (BoW) representations of the text data.
For each classifier (Decision Tree, Random Forest, LinearSVC, Multinomial Naive Bayes):

The classifier is created and trained on the vectorized training data.
Predictions are made on the test data.
Various classification metrics are calculated, including F1 score, accuracy, precision, recall, ROC curve, and AUC.
The results are printed for each classifier.
Bag of Words Vectorization and Model Evaluation:

Similar to TF-IDF vectorization, Bag of Words (BoW) vectorization is applied to the text data.
The same four classifiers are trained and evaluated on the BoW representations, and the results are printed.
Doc2Vec Vectorization and Model Evaluation:

Doc2Vec vectorization is applied to the text data using the Gensim library. It transforms the text documents into dense numerical vectors.
The code trains a Doc2Vec model on the combined dataset (training and test) and then uses this model to infer vectors for both training and test data.
Two classifiers, Support Vector Classifier (SVC) and Multinomial Naive Bayes, are trained and evaluated on the Doc2Vec vectors.
Results are printed for both classifiers.

