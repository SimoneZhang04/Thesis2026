import os
import time
import joblib
import numpy
import pandas
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

import main_tabulardata

# OPTIONAL VARS
SOURCE_FOLDER = 'calculate_difficulty'
TEST_FOLDER = os.path.join(SOURCE_FOLDER, 'test')
TRAIN_FOLDER = os.path.join(SOURCE_FOLDER, 'train')
SUB_FOLDER = 'other_datasets'
FILE_NAME = 'MetroPT2_shuffled.csv'

# NEEDED VARS
UTILS_FOLDER = 'utils'
INPUT_CLASSIFIER_FOLDER = 'input_classifier'
DIFFICULTY_COLUMN = 'difficulty_function'
PREDICTED_DIFFICULTY_COLUMN = 'predicted_difficulty_function'
DISTANCE_COLUMN = 'distance'
CONFIDENCE_COLUMN = 'confidence'
TIME_COLUMN = 'time'
NEAREST_NEIGHBOUR = os.path.join(UTILS_FOLDER, SUB_FOLDER, '%snearest_neighbour_joblib' % FILE_NAME)
SCALER = os.path.join(UTILS_FOLDER, SUB_FOLDER, '%s_scaler.joblib' % FILE_NAME)
FULL_SOURCE_FILE_NAME =os.path.join(TEST_FOLDER, SUB_FOLDER, FILE_NAME)
FULL_TRAIN_FILE_NAME = os.path.join(TRAIN_FOLDER, SUB_FOLDER, FILE_NAME)

def calculate_new_input_difficulty(source_dataframe, train_dataframe, number_of_neighbours, data_scaler, weighted):

    # Fills NAN to 0
    train_dataframe = train_dataframe.fillna(0)
    source_dataframe = source_dataframe.fillna(0)
    # Filters out columns that are not in the original (e.g. filters out score function columns)
    train_df_filtered = train_dataframe.drop(columns=column_to_remove)
    # Filters out columns that are not in the original, this time with the addition of predicted difficulty and confidence columns
    source_df_filtered = source_dataframe[train_df_filtered.columns]

    # Filters out non numeric columns
    train_df_non_numeric = train_df_filtered.select_dtypes(exclude=['object']).to_numpy()
    source_df_non_numeric = source_df_filtered.select_dtypes(exclude=['object']).to_numpy()

    train_df_scaled = data_scaler.transform(train_df_non_numeric)
    source_df_scaled = data_scaler.transform(source_df_non_numeric)
    train_df_difficulty_values = train_dataframe[DIFFICULTY_COLUMN].to_numpy()

    # Loads the NearestNeighbour model or creates it using training set
    if os.path.isfile(NEAREST_NEIGHBOUR):
        knn = joblib.load(NEAREST_NEIGHBOUR)
    else:
        knn = NearestNeighbors(n_neighbors=number_of_neighbours, algorithm= 'auto')
        knn.fit(train_df_scaled)
        joblib.dump(knn, NEAREST_NEIGHBOUR)

    distances, indices = knn.kneighbors(source_df_scaled)

    neighbors_difficulty_values = train_df_difficulty_values[indices]

    if number_of_neighbours == 1:
         # If 1 neighbour confidence is calculated from the distance from the datapoint
        confidence = (numpy.exp(-distances[:, 0]))
        difficulty = neighbors_difficulty_values[:, 0]
    elif weighted:
        # Weight for the average is the inverse of the distance, added 0.0001 to avoid division by 0
        weights = 1 / (distances + 0.0001)

        weighted_mean = numpy.average(neighbors_difficulty_values, weights=weights, axis=1)
        difficulty = weighted_mean
        # Calculate weighted variance
        weighted_variance = numpy.average((neighbors_difficulty_values - weighted_mean[:, None]) ** 2, weights=weights)

        value_std = numpy.sqrt(weighted_variance)

        mean_distance = numpy.average(distances, weights=weights, axis=1)

        uncertainty = value_std + 0.5 * mean_distance
        confidence = numpy.exp(-uncertainty)
    else:
        difficulty = numpy.average(neighbors_difficulty_values, axis=1)
        mean_distance = numpy.mean(distances, axis=1)
        value_std = numpy.std(neighbors_difficulty_values, axis=1)

        uncertainty = value_std + 0.5 * mean_distance
        confidence = numpy.exp(-uncertainty)

    suffix = '_with_' + number_of_neighbours.__str__() + 'neighbours'

    if weighted:
        suffix += '_weighted'
    source_dataframe[PREDICTED_DIFFICULTY_COLUMN + suffix] = difficulty
    source_dataframe[CONFIDENCE_COLUMN + suffix] = confidence

    return source_dataframe

# Calculates the difficulty taking the max value instead of an average
def calculate_new_input_difficulty_max_neighbours(source_dataframe, train_dataframe, number_of_neighbours, data_scaler):

    # Fills NAN to 0
    train_dataframe = train_dataframe.fillna(0)
    source_dataframe = source_dataframe.fillna(0)
    # Filters out columns that are not in the original (e.g. filters out score function columns)
    train_df_filtered = train_dataframe.drop(columns=column_to_remove)
    # Filters out columns that are not in the original, this time with the addition of predicted difficulty and confidence columns
    source_df_filtered = source_dataframe[train_df_filtered.columns]

    # Filters out non numeric columns
    train_df_non_numeric = train_df_filtered.select_dtypes(exclude=['object']).to_numpy()
    source_df_non_numeric = source_df_filtered.select_dtypes(exclude=['object']).to_numpy()

    train_df_scaled = data_scaler.transform(train_df_non_numeric)
    source_df_scaled = data_scaler.transform(source_df_non_numeric)
    train_df_difficulty_values = train_dataframe[DIFFICULTY_COLUMN].to_numpy()

    # Loads the NearestNeighbour model or creates it using training set
    if os.path.isfile(NEAREST_NEIGHBOUR):
        knn = joblib.load(NEAREST_NEIGHBOUR)
    else:
        knn = NearestNeighbors(n_neighbors=number_of_neighbours, algorithm='auto')
        knn.fit(train_df_scaled)
        joblib.dump(knn, NEAREST_NEIGHBOUR)

    distances, indices = knn.kneighbors(source_df_scaled)

    neighbors_difficulty_values = train_df_difficulty_values[indices]

    if number_of_neighbours == 1:
         # If 1 neighbour confidence is calculated from the distance from the datapoint
        confidence = (numpy.exp(-distances[:, 0]))
        difficulty = neighbors_difficulty_values[:, 0]
    else:
        difficulty = numpy.max(neighbors_difficulty_values, axis=1)
        mean_distance = numpy.mean(distances, axis=1)
        value_std = numpy.std(neighbors_difficulty_values, axis=1)

        uncertainty = value_std + 0.5 * mean_distance
        confidence = numpy.exp(-uncertainty)

    suffix = '_with_' + number_of_neighbours.__str__() + 'neighbours_MAX'
    source_dataframe[PREDICTED_DIFFICULTY_COLUMN + suffix] = difficulty
    source_dataframe[CONFIDENCE_COLUMN + suffix] = confidence

    return source_dataframe

# Calculates the difficulty by taking the one with the highest confidence from the calculated ones
def calculate_new_input_difficulty_confidence(source_dataframe):

    difficulty = []
    confidence = []
    columns = source_dataframe.filter(like=CONFIDENCE_COLUMN).columns.tolist()

    for row in source_dataframe:
        # Gets the name of the column with the highest confidence
        max_confidence_column = row[columns].idxmax(axis=1)
        # Gets the suffix of the column with the highest confidence
        column_suffix = max_confidence_column.split('_with_')[-1]

        dif = row[PREDICTED_DIFFICULTY_COLUMN + column_suffix]
        confid = row[CONFIDENCE_COLUMN + column_suffix]

        difficulty.append(dif)
        confidence.append(confid)

    suffix = '_with_' + 'confidence'

    source_dataframe[PREDICTED_DIFFICULTY_COLUMN + suffix] = difficulty
    source_dataframe[CONFIDENCE_COLUMN + suffix] = confidence

    return source_dataframe

def calculate_new_input_with_rf(source_dataframe, train_dataframe, number_of_trees):

    # Filters out columns that are not in the original (e.g. filters out score function columns), and non-numeric ones
    x_train_df_filtered = train_dataframe.drop(columns=column_to_remove).select_dtypes(exclude=['object'])
    y_train_df_filtered = train_dataframe[DIFFICULTY_COLUMN].to_numpy()
    x_source_df_filtered = source_dataframe[x_train_df_filtered.columns]
    x_source_df_filtered = x_source_df_filtered.fillna(0)
    x_source_df_filtered = x_source_df_filtered.to_numpy()
    x_train_df_filtered = x_train_df_filtered.fillna(0)
    x_train_df_filtered = x_train_df_filtered.to_numpy()

    # Initiate and train the model
    rf_model = RandomForestRegressor(n_estimators=number_of_trees, random_state=42)
    rf_model.fit(x_train_df_filtered, y_train_df_filtered)

    predictions = rf_model.predict(x_source_df_filtered)

    # Calculate the confidence
    tree_predictions = numpy.array([tree.predict(x_source_df_filtered ) for tree in rf_model.estimators_])

    prediction_sdt = numpy.std(tree_predictions, axis=0)

    confidence = numpy.exp(-prediction_sdt)

    source_dataframe[PREDICTED_DIFFICULTY_COLUMN + '_with_rf'] = predictions
    source_dataframe[CONFIDENCE_COLUMN + '_with_rf'] = confidence

    return source_dataframe, rf_model


def calculate_new_input_with_lr(source_dataframe, train_dataframe, data_scaler):

    # Filters out columns that are not in the original (e.g. filters out score function columns)
    x_train_df_filtered = train_dataframe.drop(columns=column_to_remove).select_dtypes(exclude=['object'])
    y_train_df_filtered = train_dataframe[DIFFICULTY_COLUMN].to_numpy()
    x_source_df_filtered = source_dataframe[x_train_df_filtered.columns]
    x_source_df_filtered = x_source_df_filtered.fillna(0)
    x_source_df_filtered = x_source_df_filtered.to_numpy()
    x_train_df_filtered = x_train_df_filtered.fillna(0)
    x_train_df_filtered = x_train_df_filtered.to_numpy()

    x_train_scaled = data_scaler.transform(x_train_df_filtered)
    x_source_scaled = data_scaler.transform(x_source_df_filtered)

    # Initiate and train the model
    lr_model = LinearRegression()
    lr_model.fit(x_train_scaled, y_train_df_filtered)

    predictions = lr_model.predict(x_source_scaled)

    mean = numpy.mean(x_train_scaled, axis=0)
    std = numpy.std(x_train_scaled, axis=0) + 0.0001
    z_scores = (x_source_scaled - mean) / std
    distance = numpy.linalg.norm(z_scores, axis=1)

    confidence = numpy.exp(-distance)
    source_dataframe[PREDICTED_DIFFICULTY_COLUMN + '_with_lr'] = predictions
    source_dataframe[CONFIDENCE_COLUMN + '_with_lr'] = confidence

    return source_dataframe, lr_model

if __name__ == '__main__':
    column_to_remove = []
    # Reads source df
    source_df = pandas.read_csv(FULL_SOURCE_FILE_NAME, sep=",")
    # Reads df with their domains and difficulty
    train_df = pandas.read_csv(FULL_TRAIN_FILE_NAME, sep=",")
    # Loads scaler, same one as the one used to scale data to give to classifier
    scaler = joblib.load(SCALER)

    for classifier in main_tabulardata.get_learners(0.5):
        column_to_remove.append("%s_score" % type(classifier).__name__)

    column_to_remove.append(DIFFICULTY_COLUMN)

    start_time = time.time()
    df_final = calculate_new_input_difficulty(source_df.copy(), train_df.copy(), 3, scaler, weighted=True)
    end_time = time.time()
    total_time = end_time - start_time
    time_for_row = total_time / len(df_final)
    suffix = df_final.columns[-1].split(CONFIDENCE_COLUMN)[-1]
    df_final[TIME_COLUMN + suffix] = time_for_row * 1000

    # Saves the df with calculated difficulty to .csv file
    df_final.to_csv(FULL_SOURCE_FILE_NAME, index=False, float_format='%.5f')



    # REPEATS FOR 5 TIMES WITH DIFFERENT ALGORITHMS/PARAMETER TO DELETE AFTER
    # REPEATS FOR 5 TIMES WITH DIFFERENT ALGORITHMS/PARAMETER TO DELETE AFTER
    # REPEATS FOR 5 TIMES WITH DIFFERENT ALGORITHMS/PARAMETER TO DELETE AFTER

    source_df = pandas.read_csv(FULL_SOURCE_FILE_NAME, sep=",")
    start_time = time.time()
    df_final, model = calculate_new_input_with_rf(source_df.copy(), train_df.copy(), 100)
    end_time = time.time()
    total_time = end_time - start_time
    time_for_row = total_time / len(df_final)
    suffix = df_final.columns[-1].split(CONFIDENCE_COLUMN)[-1]
    df_final[TIME_COLUMN + suffix] = time_for_row * 1000
    # Saves the df with calculated difficulty to .csv file
    df_final.to_csv(FULL_SOURCE_FILE_NAME, index=False, float_format ='%.5f')
    dump( model, os.path.join(UTILS_FOLDER,INPUT_CLASSIFIER_FOLDER, "%s_%s.joblib" % (FILE_NAME, type(model).__name__)))

    source_df = pandas.read_csv(FULL_SOURCE_FILE_NAME, sep=",")
    start_time = time.time()
    df_final, model = calculate_new_input_with_lr(source_df.copy(), train_df.copy(), scaler)
    end_time = time.time()
    total_time = end_time - start_time
    time_for_row = total_time / len(df_final)
    suffix = df_final.columns[-1].split(CONFIDENCE_COLUMN)[-1]
    df_final[TIME_COLUMN + suffix] = time_for_row * 1000
    # Saves the df with calculated difficulty to .csv file
    df_final.to_csv(FULL_SOURCE_FILE_NAME, index=False, float_format='%.5f')
    dump(model, os.path.join(UTILS_FOLDER, INPUT_CLASSIFIER_FOLDER, "%s_%s.joblib" % (FILE_NAME, type(model).__name__)))

    source_df = pandas.read_csv(FULL_SOURCE_FILE_NAME, sep=",")
    start_time = time.time()
    df_final= calculate_new_input_difficulty(source_df.copy(), train_df.copy(), 3, scaler, weighted=False)
    end_time = time.time()
    total_time = end_time - start_time
    time_for_row = total_time / len(df_final)
    suffix = df_final.columns[-1].split(CONFIDENCE_COLUMN)[-1]
    df_final[TIME_COLUMN + suffix] = time_for_row * 1000
    # Saves the df with calculated difficulty to .csv file
    df_final.to_csv(FULL_SOURCE_FILE_NAME, index=False, float_format='%.5f')

    source_df = pandas.read_csv(FULL_SOURCE_FILE_NAME, sep=",")
    start_time = time.time()
    df_final = calculate_new_input_difficulty(source_df.copy(), train_df.copy(), 5, scaler, weighted=True)
    end_time = time.time()
    total_time = end_time - start_time
    time_for_row = total_time / len(df_final)
    suffix = df_final.columns[-1].split(CONFIDENCE_COLUMN)[-1]
    df_final[TIME_COLUMN + suffix] = time_for_row * 1000
    # Saves the df with calculated difficulty to .csv file
    df_final.to_csv(FULL_SOURCE_FILE_NAME, index=False, float_format='%.5f')

    source_df = pandas.read_csv(FULL_SOURCE_FILE_NAME, sep=",")
    start_time = time.time()
    df_final = calculate_new_input_difficulty(source_df.copy(), train_df.copy(), 5, scaler, weighted=False)
    end_time = time.time()
    total_time = end_time - start_time
    time_for_row = total_time / len(df_final)
    suffix = df_final.columns[-1].split(CONFIDENCE_COLUMN)[-1]
    df_final[TIME_COLUMN + suffix] = time_for_row * 1000
    # Saves the df with calculated difficulty to .csv file
    df_final.to_csv(FULL_SOURCE_FILE_NAME, index=False, float_format='%.5f')
