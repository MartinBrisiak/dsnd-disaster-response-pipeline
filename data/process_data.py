import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the data from csv files into pandas dataframe and merge them based on "id" column.

    :param messages_filepath: path to messages csv file
    :param categories_filepath: path to categories csv file
    :return: merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, how="left", on="id")


def clean_data(df):
    """
    Cleans the input messages and categories by:
    - splitting categories into multiple columns
    - transforming categories from string of text "related-1" to integer "1"

    :param df: dataframe of messages with categories in unclean form
    :return: dataframe of messages with categories in clean form
    """
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: int(x.split("-")[1]))
    df = df.drop(labels=["categories"], axis=1)
    df = pd.concat([df, categories], axis=1)
    return df.drop_duplicates(keep=False)


def save_data(df, database_filename):
    """
    Saves the dataframe of messages and categories into the sql lite database with the table "messages".

    :param df: cleaned dataframe of messages with categories
    :param database_filename: name of the file to save the database into
    :return: nothing
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
