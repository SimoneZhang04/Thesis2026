Repository used for the code used in the thesis, where we try to find the difficulty of input in classifiers. 

main_tabulardata.py trains and saves the classifier with scaler for each .csv file in a folder.

calculate_difficulty_function.py uses the saved classifier to calculate the difficulty function of each datapoint in each .csv file in a folder, by doing a mean of the score function of each classifier ( 0 if the classification is right, 1 if wrong). The output is saved in a .csv file in the folder called difficulty.

The folder where the .csv file are taken from is specified by the global variable CSV_FOLDER in main_tabulardata.py.

For now the classifier used are simple ones, to create working code later more will be added to get theoretic results.

