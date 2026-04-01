Classifier Input Difficulty Analysis
This repository contains the code used for my thesis research on quantifying the difficulty of data inputs.

Project Overview
The goal of this project is to determine how "difficult" a specific datapoint is for any given classifier. A score function is defined for each classifier where:

0: Correct classification

1: Incorrect classification

The difficulty is the mean of these scores across all trained classifiers.

Core Scripts

1. main_tabulardata.py
Purpose: Iterates through all .csv files in the source folder to train classifiers, also saves the score of each classifier in a .csv file.

Outputs: The trained models and their respective scaler objects.

Configuration: <br />
CSV_FOLDER = Folder where the .csv file to work on are saved. <br />
LABEL_NAME = Name of the label column of the .csv file. Assumed to be the same for all files in the folder. <br />
NORMAL_TAG = String rappresenting the normal situation. <br />
SCORES_FILE = File where the score for each classifier will be saved. <br />
TT_SPLIT = Percentage of the train-test split. (e.g. 0.3 is 30% test, 70% train) <br />
VERBOSE = True if debug information needs to be shown. <br />
BINARIZE = True if we want to conduct anomaly detection, transforms multi-class labels into binary labels. <br />
UTILS_FOLDER = Folder where the utilities will be saved (Classifiers, scalers, etc.). Files will be saved in a sub-folder with the same name as CSV_FOLDER. <br />

2. calculate_difficulty_function.py
Purpose: Loads the saved models and calculates the difficulty for every datapoint in every .csv file of a folder.

Output: Generates new CSV files in the /difficulty sub-folder containing the original data plus the calculated difficulty scores.

Configuration: <br />
SOURCE_FOLDER = Folder where the .csv file to work on are saved. <br />
OUTPUT_FOLDER = Folder where the outpuit will be saved. <br />
UTILS_FOLDER = Folder where the utilities are saved (Classifiers, scalers, etc.).  <br />
DIFFICULTY_FUNCTION_COL = Name of the difficulty-function column of the .csv file. Assumed to be the same for all files in the folder. <br />

3. divide_file_to_train_&_test.py  
Purpose: Divide the files with difficulty-function obtained from calculate_difficulty_function.py, into 2 separate files with the train_test_split() function from the sklearn library, stratifying with multilabel if avaliable, otherwhise with label.

Output: Generates 2 .csv files, into 2 separate folders (train, test) with the same name as the original, for each .csv file in the folder.

Configuration: <br />
SOURCE_FOLDER = Path of your source folder. <br />
OUTPUT_FOLDER = Path of the folder where the output files will be saved, divided in 2 sub-folders. <br />
TEST_FOLDER = Sub-folder of the OUTPUT_FOLDER, where the test split will be saved. <br />
TRAIN_FOLDER = Sub-folder of the OUTPUT_FOLDER, where the train split will be saved. <br />
MULTILABEL_COLUMN = Name of the multilabel column of the .csv files. Assumed to be the same for all files in the folder. <br />
LABEL_COLUMN =  Name of the label column of the .csv files. Assumed to be the same for all files in the folder. <br />
SPLIT_SIZE = Percentage of the train-test split (e.g. 0.3 is 30% test, 70% train). <br />

4. calculate_new_input_difficulty.py
Purpose: Calculate the difficulty function of a new datapoint with its confidence of the given value, using different methods:
calculate_new_input_difficulty: Calculates the difficulty based on the nearest neighbours, with the option of doing it with weight. <br />
calculate_new_input_with_rf: Calculates the difficulty with a RandomForestRegressor model. <br />
calculate_new_input_with_lr: Calculates the difficulty with a LinearRegressor model. <br />
calculate_new_input_with_xgb : Calculates the difficulty with a XGB (Extreme Gradient Boosting) model. <br />
calculate_new_input_difficulty_max_neighbours: Calculates the difficulty by taking the highest diffficulty among its neighbours. <br />
calculate_new_input_difficulty_confidence:  Calculates the difficulty by taking the difficulty of the previous prediction with the highest confidence. <br />

Output: Generates a .csv file where each datapoint has the calculated difficulty with its confidence, and the classifier that was used if one was used.

Configuration: <br />
UTILS_FOLDER = Folder where the utilities are saved (Classifiers, scalers, etc.).  <br />
INPUT_CLASSIFIER_FOLDER = Folder where the used Classifier will be saved. <br />
DIFFICULTY_COLUMN = Name of the difficulty-function column of the training file. <br />
DISTANCE_COLUMN = Name of the distance column that will be used to calculate the distance, will not be saved to file. <br />
PREDICTED_DIFFICULTY_COLUMN = Prefix of the column that will have the predicted-difficulty-function, will be saved to the file. <br />
CONFIDENCE_COLUMN =  Prefix of the column that will have the confidence of each prediction, will be saved to the file. <br />
SCALER = Path where the scaler for the file createed from main_tabulardata.py is saved. <br />
TIME_COLUMN = Prefix of the column that will have the average time taken to calculate the difficulty, will be saved to the file. <br />
NEAREST_NEIGHBOUR = Path where the file with the NearestNeighbour file generated from sklearn.neighbors is saved. <br />
FULL_SOURCE_FILE_NAME = Path where the file with datapoint to predict the difficulty is saved. <br />
FULL_TRAIN_FILE_NAME = Path where the file with the dataset and the difficulty from calculate_difficulty_function.py is saved. <br />
LR_FILE_NAME =  Path where the file with the lr model is saved. <br />
SUB_FOLDER = Sub-folder where the different files are saved. <br />
FILE_NAME = Name of the file that is used in the loading/saving of xgb and rf models. <br />
5. calculate_MAE.py
Purpose: Calculate the MEA and average of the confidence for each method used in calculate_new_input_difficulty.py:

Output: Generates a .csv file where for each method there's a row with MEA, average confidence.

Configuration: <br />
DIFFICULTY_COLUMN = Name of the difficulty-function column of the file. <br />
PREDICTED_DIFFICULTY_COLUMN =  Prefix of the column that has the predicted-difficulty-function. <br />
CONFIDENCE_COLUMN = Prefix of the column that has the confidence of each prediction. <br />
TIME_COLUMN = Prefix of the column that will have the average time taken to calculate the difficulty. <br />
FULL_FILE_NAME = Path where the file to work on is saved.  <br />
FULL_OUTPUT_NAME = Path where the output will be saved.  <br />

Requirements: <br />
os  <br />
joblib <br />
numpy <br />
pandas <br />

How to run

First train the classifier with main_tabulardata.py, then use calculate_difficulty_function.py to calculate the difficulties, use calculate_new_input_difficulty.py to calculate the difficulty of a new datapoint.

TO DO
