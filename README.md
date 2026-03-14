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

Configuration: Set the CSV_FOLDER global variable to point to your source folder.

2. calculate_difficulty_function.py
Purpose: Loads the saved models and calculates the difficulty for every datapoint in the original datasets.

Output: Generates new CSV files in the /difficulty subdirectory containing the original data plus the calculated difficulty scores.


Requirements
TODO

How to run

First train the classifier with main_tabulardata.py, and then run calculate_difficulty_function.py to calculate the difficulties.

TO DO

Create a program to divide the datapoints in different domains / analyze how to create those divisions.
Create a program to calculate the difficulty of a new datapoint without doing the calculation for each classifier but doing a comparison with its domain and  closeness to another exsisting datapoint.
Clean the code/refactor it (output files have .csv in between their name etc.).
