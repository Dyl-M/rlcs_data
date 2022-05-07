# -*- coding: utf-8 -*-

import ml_formatting

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import rcParams
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

"""File Information

@file_name: ml_training.py
@author: Dylan "dyl-m" Monfret
"""

"GLOBAL"

with open('../../data/private/random_seeds.json', 'r', encoding='utf8') as seeds_file:
    SEEDS = json.load(seeds_file)

RANDOM_SEED, RANDOM_SEED_2 = SEEDS['seed_1'], SEEDS['seed_2']

"OPTIONS"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

tf.random.set_seed(RANDOM_SEED)

rcParams['figure.figsize'] = (19.2, 10.8)

"FUNCTIONS"


def pretreatment(original_data: pd.DataFrame, split_data: pd.DataFrame):
    """Prepare raw data for further treatments
    :param original_data: original dataframe for fit step application and to get columns by type
    :param split_data: split / sample data resulting from sklearn 'train_test_split' or K-folds operation
    :return split_data_final: formatted data as numpy array.
    """
    num_cols = original_data.select_dtypes(include=np.number).columns.to_list()  # Getting numerical columns
    cat_cols = original_data.select_dtypes(exclude=np.number).columns.to_list()  # Getting categorical columns

    split_data_cat = OneHotEncoder(drop='if_binary', handle_unknown='ignore') \
        .fit(original_data[cat_cols]) \
        .transform(split_data[cat_cols]) \
        .toarray()  # From Categorical to One Hot

    split_data_scaled = StandardScaler().fit(original_data[num_cols]).transform(split_data[num_cols])  # Standardisation
    split_data_final = np.concatenate((split_data_cat, split_data_scaled), axis=1)  # Concatenation

    return split_data_final


def compile_model(train: np.array, train_target: np.array, validation: np.array, val_target: np.array,
                  batch_size: float, alpha: float = 0.01, es_rate: float = 0.2, epochs: int = 100, workers: int = 1,
                  verbose: bool = True):
    """Compile and fit a keras model
    :param train: train array
    :param train_target: target array
    :param validation: validation array
    :param val_target: validation target array
    :param batch_size: number of samples to work through before updating the internal model parameters
    :param alpha: initial learning rate
    :param es_rate: early stopping rate, epochs percentage to set early stopping
    :param epochs: number times that the learning algorithm will work through the entire training dataset
    :param workers: maximum number of processes to spin up when using process-based threading
    :param verbose: to display progress or not
    :return: model fitted (Keras model + Keras History).
    """
    if es_rate > 1:  # Patience can't be superior to epochs
        es_rate = 1

    elif es_rate < 0:  # Patience set to 0 is equivalent at not setting early stopping
        es_rate = 0.2

    # Early Stopping settings
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=es_rate * epochs,
                                                      mode='min',
                                                      restore_best_weights=True)

    # Alpha reducer
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                          factor=0.1,
                                                          patience=es_rate * epochs / 4,
                                                          verbose=0,
                                                          mode='min',
                                                          min_lr=1e-6)

    # Model Checkpoint

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='../../models/tmp_mdl.hdf5',
                                                    monitor='val_accuracy',
                                                    save_weights_only=True,
                                                    mode='max',
                                                    save_best_only=True)

    # Layers
    model = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(256, activation='relu'),
                                 tf.keras.layers.Dense(256, activation='relu'),
                                 tf.keras.layers.Dense(1, activation='sigmoid')])

    # Compilation
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),
                  metrics=tf.keras.metrics.BinaryAccuracy(name='accuracy'))

    # Fit
    history = model.fit(train,
                        train_target,
                        callbacks=[early_stopping, reduce_lr_loss, checkpoint],
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(validation, val_target),
                        verbose=verbose,
                        workers=workers)

    # Reloading weights
    model.load_weights('../../models/tmp_mdl.hdf5')

    return model, history


def plot_history(model_history: tf.keras.callbacks.History, batch_size: int, alpha: float, export: bool = False,
                 display: bool = False):
    """Plot fit history, logistic losses and accuracies
    :param model_history: a Keras model
    :param batch_size: batch size, number of samples to work through before updating the internal model parameters
    :param alpha: initial learning rate
    :param export: to export plot or not
    :param display: to display / show plot or not
    """
    train_len = len(model_history.history['loss'])
    epoch_set = model_history.params['epochs']

    fig, ax_1 = plt.subplots()

    ax_1.set_xlabel('Epoch', size=14)
    ax_1.set_ylabel('Log Loss', size=14)

    loss = ax_1.plot(np.arange(1, train_len + 1),
                     model_history.history['loss'],
                     label='Loss - Train Set',
                     color='#D81B60')

    val_loss = ax_1.plot(np.arange(1, train_len + 1),
                         model_history.history['val_loss'],
                         label='Loss - Validation Set',
                         color='#FFC107')

    ax_1.tick_params(axis='y')

    ax_2 = ax_1.twinx()
    ax_2.set_ylabel('Accuracy', size=14)

    acc = ax_2.plot(np.arange(1, train_len + 1),
                    model_history.history['accuracy'],
                    label='Accuracy - Train Set',
                    color='#004D40')

    val_ac = ax_2.plot(np.arange(1, train_len + 1),
                       model_history.history['val_accuracy'],
                       label='Accuracy - Validation Set',
                       color='#1E88E5')

    ax_2.tick_params(axis='y')

    all_series = loss + val_loss + acc + val_ac
    labels = [series.get_label() for series in all_series]
    ax_1.legend(all_series, labels, prop={'size': 15})

    ax_1.grid(axis='x')

    plt.suptitle('Losses & Accuracies evolutions', size=20)
    plt.title(f'Epochs set: {epoch_set} | Training length: {train_len} | Batch size: {batch_size} | Init. Alpha:'
              f' {alpha:.0e}', size=16)

    fig.tight_layout()

    if display:
        plt.show()

    if export:
        plt.savefig(f'../../reports/figures/model batch_{batch_size} lr_{alpha:.0e}.jpg')

    plt.close()


def get_predictions(model: tf.keras.Model, test: np.array, test_target: np.array, batch_size, alpha):
    """Do predictions: return probabilities, classes and scores
    :param model: fitted model
    :param test: test array
    :param test_target: test target
    :param batch_size: batch size, number of samples to work through before updating the internal model parameters
    :param alpha: initial learning rate
    :return results: dictionary with probabilities, classes and accuracy, precision and recall scores.
    """
    results = {}
    predictions = model.predict(test)
    prediction_classes = np.ravel(predictions).round()
    probabilities = np.concatenate((1 - np.vstack(predictions), predictions), axis=1)

    results['batch_size'] = batch_size
    results['init_alpha'] = alpha
    results['log_loss'] = log_loss(test_target, probabilities)
    results['accuracy'] = accuracy_score(test_target, prediction_classes)
    results['f1_score'] = f1_score(test_target, prediction_classes)

    print(f'TEST SET EVALUATION (batch={batch_size}, init. alpha={alpha:.0e})\n'
          f' > Log Loss: {results["log_loss"]:.4f}\n'
          f' > Accuracy: {results["accuracy"]:.4f}\n'
          f' > F1-Score: {results["f1_score"]:.4f}\n')

    return results


def model_tuning(x: np.array, y: np.array, epochs: int, es_rate: float, batch_grid: list, alpha_grid: list,
                 workers: int = 1, verbose: bool = False, export_graph: bool = True, display_graph: bool = False):
    """Test learning rate and batch size values with Keras model implemented in 'compile_model' function
    :param x: training instances to class
    :param y: target array relative to x
    :param epochs: number times that the learning algorithm will work through the entire training dataset
    :param es_rate: early stopping rate, epochs percentage to set early stopping
    :param batch_grid: list of batch size to test
    :param alpha_grid: list of learning rate to test
    :param workers: maximum number of processes to spin up when using process-based threading
    :param verbose: to display progress or not
    :param export_graph: to export plots or not
    :param display_graph: to display / show plots or not
    :return predictions_results: metrics on test set predictions
    """

    def perf(value: float, ref_val: float, mode_max: bool = True):
        """Compute "model performance"
        :param value: comparison value
        :param ref_val: reference value (a max or a min)
        :param mode_max: if reference value is max or not
        :return: performance score.
        """
        if mode_max:
            return value * 100 / ref_val
        return ref_val * 100 / value

    predictions_results = []
    param_grid = [(b, a) for b in batch_grid for a in alpha_grid]

    # Split Train / Validation / Test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=RANDOM_SEED, stratify=y)

    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=1 / 3,
                                                      random_state=RANDOM_SEED,
                                                      stratify=y_train)

    # Pretreatments
    new_x_train = pretreatment(original_data=x, split_data=x_train)
    new_x_val = pretreatment(original_data=x, split_data=x_val)
    new_x_test = pretreatment(original_data=x, split_data=x_test)

    for idx, set_of_par in enumerate(param_grid):
        print(f'Model # {idx + 1} out of {len(param_grid)}')
        # Model compilation
        model, model_history = compile_model(train=new_x_train,
                                             train_target=y_train,
                                             validation=new_x_val,
                                             val_target=y_val,
                                             batch_size=set_of_par[0],
                                             alpha=set_of_par[1],
                                             epochs=epochs,
                                             es_rate=es_rate,
                                             verbose=verbose,
                                             workers=workers)

        # Predictions & Scoring
        predictions_results.append(get_predictions(model=model,
                                                   test=new_x_test,
                                                   test_target=y_test,
                                                   batch_size=set_of_par[0],
                                                   alpha=set_of_par[1]))

        if export_graph or display_graph:
            # Plot
            plot_history(model_history,
                         batch_size=set_of_par[0],
                         alpha=set_of_par[1],
                         export=export_graph,
                         display=display_graph)

    # Compare results
    evaluation_df = pd.DataFrame(predictions_results)
    min_log_loss = evaluation_df.log_loss.min()
    max_accuracy = evaluation_df.accuracy.max()
    max_accuracy = evaluation_df.accuracy.max()
    max_f1_score = evaluation_df.f1_score.max()
    evaluation_df['log_loss_perf'] = evaluation_df.log_loss.apply(lambda val: perf(val, min_log_loss, mode_max=False))
    evaluation_df['accuracy_perf'] = evaluation_df.accuracy.apply(lambda val: perf(val, max_accuracy))
    evaluation_df['f1_score_perf'] = evaluation_df.f1_score.apply(lambda val: perf(val, max_f1_score))
    evaluation_df['mean_perf'] = evaluation_df.loc[:, ['log_loss_perf', 'accuracy_perf', 'f1_score_perf']].mean(axis=1)
    evaluation_df = evaluation_df.sort_values('mean_perf', ascending=False).reset_index(drop=True)
    evaluation_df.batch_size = evaluation_df.batch_size.astype(int)
    best_settings = evaluation_df[['batch_size', 'init_alpha']].iloc[0, :].to_dict()

    # Export results
    evaluation_df.to_csv('../../models/tuning_results/evaluation.csv', encoding='utf8', index=False)

    return evaluation_df, best_settings


def compile_best_model(x: np.array, y: np.array, epochs: int, es_rate: float, batch_size: int, alpha: float,
                       workers: int = 1, verbose: bool = True):
    """Compile best model (the best combination of batch size & alpha) with Keras model implemented in 'compile_model'
    function
    :param x: training instances to class
    :param y: target array relative to x
    :param epochs: number times that the learning algorithm will work through the entire training dataset
    :param es_rate: early stopping rate, epochs percentage to set early stopping
    :param batch_size: batch size, number of samples to work through before updating the internal model parameters
    :param alpha: initial learning rate
    :param workers: maximum number of processes to spin up when using process-based threading
    :param verbose: to display progress or not
    :return model, model_history: Keras model and model's history
    """
    # Split Train / Validation / Test
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=1 / 3, random_state=RANDOM_SEED_2, stratify=y)

    # Pretreatments
    new_x_train = pretreatment(original_data=x, split_data=x_train)
    new_x_val = pretreatment(original_data=x, split_data=x_val)

    # Model compilation
    model, model_history = compile_model(train=new_x_train,
                                         train_target=y_train,
                                         validation=new_x_val,
                                         val_target=y_val,
                                         batch_size=int(batch_size),
                                         alpha=alpha,
                                         epochs=epochs,
                                         es_rate=es_rate,
                                         verbose=verbose,
                                         workers=workers)

    return model, model_history


"MAIN"

if __name__ == '__main__':
    # Data import and formatting
    DF_GAMES = ml_formatting.treatment_by_players()

    # Extract target array
    DATA = DF_GAMES.drop('winner', axis=1)
    TARGET = DF_GAMES.winner

    # Tuning Settings
    EPOCHS = 500
    BATCH_GRID = [32, 64, 128, 256]
    ALPHA_GRID = [1e-3, 1e-4, 1e-5, 1e-6]

    # Model tuning & Report
    EVALUATION_DF, BEST_SETTINGS = model_tuning(x=DATA,
                                                y=TARGET,
                                                epochs=EPOCHS,
                                                es_rate=0.10,
                                                batch_grid=BATCH_GRID,
                                                alpha_grid=ALPHA_GRID,
                                                workers=12)

    print(f'BEST SETTINGS: {BEST_SETTINGS}')

    # Training and save optimal model
    REF_MODEL, BEST_MODEL_HISTORY = compile_best_model(x=DATA, y=TARGET, epochs=1000, es_rate=0.10,
                                                       batch_size=BEST_SETTINGS['batch_size'],
                                                       alpha=BEST_SETTINGS['init_alpha'], workers=12)

    REF_MODEL.save('../../models/best_model.h5')
