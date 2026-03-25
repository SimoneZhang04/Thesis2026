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

Outputs: Saves the trained models and their respective scaler objects, in the source folder.

Configuration: <br />
CSV_FOLDER = Folder where the .csv file to work on are saved. <br />
LABEL_NAME = Name of the label column of the .csv file. Assumed to be the same for all files in the folder. <br />
NORMAL_TAG = String rappresenting the normal situation. <br />
SCORES_FILE = File where the score for each classifier will be saved. <br />
TT_SPLIT = Percentage of the train-test split. (e.g. 0.3 is 30% test, 70% train) <br />
VERBOSE = True if debug information needs to be shown. <br />
BINARIZE = True if we want to conduct anomaly detection, transforms multi-class labels into binary labels. <br />
UTILS_FOLDER = Folder where the utilities will be saved (Classifiers, scalers, etc.). <br />

2. calculate_difficulty_function.py
Purpose: Loads the saved models and calculates the difficulty for every datapoint in every .csv file of a folder.

Output: Generates new CSV files in the /difficulty sub-folder containing the original data plus the calculated difficulty scores.

Configuration: <br />
SOURCE_FOLDER = Folder where the .csv file to work on are saved. <br />
OUTPUT_FOLDER = Folder where the outpuit will be saved. <br />
UTILS_FOLDER = Folder where the utilities are saved (Classifiers, scalers, etc.).  <br />
DIFFICULTY_FUNCTION_COL = Name of the difficulty-function column of the .csv file. Assumed to be the same for all files in the folder. <br />

3. calculate_difficulty_function.py
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

Requirements
TODO

How to run

First train the classifier with main_tabulardata.py, and then run calculate_difficulty_function.py to calculate the difficulties.

TO DO
