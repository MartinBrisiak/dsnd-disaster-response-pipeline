import sys
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Loads the dataframe from "messages" table of sqllite database.
    For the data training input part there is messages column and for the lables there are categories.

    :param database_filepath: name of the file of sql lite db
    :return: training data, labels and categories
    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("messages", engine)
    X = df["message"]
    categories = ['related', 'request', 'offer',
                  'aid_related', 'medical_help', 'medical_products',
                  'search_and_rescue', 'security', 'military', 'child_alone', 'water',
                  'food', 'shelter', 'clothing', 'money', 'missing_people',
                  'refugees', 'death', 'other_aid', 'infrastructure_related',
                  'transport', 'buildings', 'electricity', 'tools', 'hospitals',
                  'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
                  'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather',
                  'direct_report']
    Y = df[categories]
    return X, Y, categories


def tokenize(text):
    """
    Tokenize and then lemmatize the training data.

    :param text: sentence to be messed up
    :return: messed up sentence
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(token).lower().strip() for token in tokens]


def build_model():
    """
    Creates the model out of transformers:
     - count vectorizer
     - Tfidf transformer
    and classifiers:
     - combined multi output classifier of random forest
    Model is then parametrized with grid search.

    :return: the model!
    """

    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'text_pipeline__vect__max_df': (0.75, 1.0),
        'text_pipeline__vect__max_features': (None, 10000),
        'text_pipeline__tfidf__use_idf': (True, False)
    }
    return GridSearchCV(pipeline, param_grid=parameters, verbose=10, n_jobs=2)


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model based on prediction and testing labels.
    Fot every category there is accuracy, precision, recall, and f1 scores calculated.

    :param model: the model!
    :param X_test: testing data
    :param Y_test: testing labels
    :param category_names: nope
    :return: nothing
    """

    y_pred = model.predict(X_test)
    Y_test_as_array = np.array(Y_test)
    for i in range(len(category_names)):
        print("{} accuracy {} precision {} recall {} f1 {}".format(
            category_names[i],
            (y_pred[:, i] == Y_test_as_array[:, i]).mean(),  # accuracy
            precision_score(Y_test_as_array[:, i], y_pred[:, i], average=None),  # precision
            recall_score(Y_test_as_array[:, i], y_pred[:, i], average=None),  # recall
            f1_score(Y_test_as_array[:, i], y_pred[:, i], average=None)  # f1
        ))
    print("mean accuracy {}".format((y_pred == Y_test_as_array).mean().mean()))


def save_model(model, model_filepath):
    """
    Saves the model as pickle file.

    :param model: the model!
    :param model_filepath: path to save the model
    :return: nothing
    """

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_filepath):
    """
    for dev purposes
    :param model_filepath:
    :return:
    """
    with open(model_filepath, 'rb') as f:
        return pickle.load(f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
