# -*- coding: utf-8 -*-

import ml_formatting

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import rcParams
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

"""File Informations

@file_name: ml_main.py
@author: Dylan "dyl-m" Monfret
"""

"GLOBAL"

RED_DATE_STR = '2022-01-23 16:00:00+00:00'  # Europe - Winter Regional 1
RANDOM_SEED = 42069

"OPTIONS"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

tf.random.set_seed(RANDOM_SEED)

rcParams['figure.figsize'] = (19.2, 10.8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

"FUNCTIONS"


def pretreatment(original_data: pd.DataFrame, split_data: pd.DataFrame):
    """Prepare raw data for further treatments
    :param original_data: original dataframe for fit step application and to get columns by type
    :param split_data: split data resulting from sklearn 'train_test_split' or K-folds operation
    :return split_data_final: formatted data as numpy array
    """
    num_cols = original_data.select_dtypes(include=np.number).columns.to_list()  # Getting numerical columns
    cat_cols = original_data.select_dtypes(exclude=np.number).columns.to_list()  # Getting categorical columns

    split_data_cat = OneHotEncoder(drop='if_binary') \
        .fit(original_data[cat_cols]) \
        .transform(split_data[cat_cols]) \
        .toarray()  # From Categorical to One Hot

    split_data_scaled = StandardScaler().fit(original_data[num_cols]).transform(split_data[num_cols])  # Standardisation

    split_data_final = np.concatenate((split_data_cat, split_data_scaled), axis=1)  # Concatenation

    return split_data_final


def compile_model(train, train_target, lr=0.05, epoch=100):
    """
    Compile the specific keras model
    :param train: train array
    :param train_target: target array
    :param lr: learning rate
    :param epoch: iterations
    :return compilation: model fitted
    """
    model = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(256, activation='relu'),
                                 tf.keras.layers.Dense(256, activation='relu'),
                                 tf.keras.layers.Dense(1, activation='sigmoid')])  # Layers

    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])  # Compilation

    history = model.fit(train, train_target, epochs=epoch)  # Fit

    return model, history


def plot_history(fitted_model):
    """
    Plot fit history
    :param fitted_model: a Keras model
    """
    plt.plot(np.arange(1, 101), fitted_model.history['loss'], label='Loss')
    plt.plot(np.arange(1, 101), fitted_model.history['accuracy'], label='Accuracy')
    plt.plot(np.arange(1, 101), fitted_model.history['precision'], label='Precision')
    plt.plot(np.arange(1, 101), fitted_model.history['recall'], label='Recall')
    plt.title('Evaluation metrics', size=20)
    plt.xlabel('Epoch', size=14)
    plt.legend()
    plt.show()


def get_predictions(model, test, test_target):
    """
    Do predictions: return probabilities, classes and scores
    :param model: fitted model
    :param test: test array
    :param test_target: test target
    :return results: dictionary with probabilities, classes and accuracy, precision and recall scores
    """
    results = {}
    predictions = model.predict(test)
    prediction_classes = np.ravel(predictions).round()

    results['probabilities'] = predictions
    results['predictions'] = prediction_classes
    results['accuracy'] = accuracy_score(test_target, prediction_classes)
    results['precision'] = precision_score(test_target, prediction_classes)
    results['recall'] = recall_score(test_target, prediction_classes)

    print(f'PREDICTIONS ON TEST\n'
          f'Accuracy: {results["accuracy"]:.2f}\n'
          f'Precision: {results["precision"]:.2f}\n'
          f'Recall: {results["recall"]:.2f}')

    return results


"MAIN"

if __name__ == '__main__':
    "Recode target variable"

    df = ml_formatting.treatment_by_teams(ref_date_str=RED_DATE_STR)
    df.overtime_seconds = df.overtime_seconds.fillna(0)
    df = df.drop('bo_id', axis=1)
    df.win = np.where(df['win'] == 'orange', 0, 1)  # Orange side is 0 and Blue side is 1

    "Split Train/Test"

    x = df.drop('win', axis=1)
    y = df.win

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)

    "Pretreatment"

    new_x_train = pretreatment(original_data=x, split_data=x_train)
    new_x_test = pretreatment(original_data=x, split_data=x_test)

    "Model conception"

    my_model, model_history = compile_model(train=new_x_train, train_target=y_train, lr=0.01)

    "Plot"

    plot_history(model_history)

    "Predictions & Scoring"

    my_predictions = get_predictions(model=my_model, test=new_x_test, test_target=y_test)
