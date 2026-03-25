# Support libs
import os
import time
import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.model_selection as ms

# Scikit-Learn algorithms
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# The PYOD library contains implementations of unsupervised classifiers.
# Works only with anomaly detection (no multi-class)
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF

# Used to save a classifier and measure its size in KB
from joblib import dump

# ------- GLOBAL VARS -----------

# Name of the folder in which look for tabular (CSV) datasets
CSV_FOLDER = "HW_Failure"
# Name of the column that contains the label in the tabular (CSV) dataset
LABEL_NAME = 'multilabel'
# Name of the 'normal' class in datasets. This will be used only for binary classification (anomaly detection)
NORMAL_TAG = 'normal'
# Name of the file in which outputs of the analysis will be saved
SCORES_FILE = "scores.csv"
# Percantage of test data wrt train data
TT_SPLIT = 0.3
# True if debug information needs to be shown
VERBOSE = True
# True if we want to conduct anomaly detection.
# This transforms multi-class labels into binary labels (rule: normal class vs others)
BINARIZE = True

UTILS_FOLDER = 'utils'
# SCALER SAVE FILE "%s_scaler.joblib" % full_name
# CLASSIFIER SAVE FILE "%s_%s.joblib" % (full_name, type(classifier).__name__)
# --------- SUPPORT FUNCTIONS ---------------


def current_milli_time():
    """
    gets the current time in ms
    :return: a long int
    """
    return round(time.time() * 1000)


def get_learners(cont_perc):
    """
    Function to get a learner to use, given its string tag
    :param cont_perc: percentage of anomalies in the training set, required for unsupervised classifiers from PYOD
    :return: the list of classifiers to be trained
    """
    learners = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        XGBClassifier(),
        MultinomialNB(),
        LinearDiscriminantAnalysis(),
        KNeighborsClassifier(n_neighbors=1),
        LogisticRegression(),
        MLPClassifier(solver='lbfgs',
                      alpha=1e-5,
                      hidden_layer_sizes=(10),
                      random_state=1),
        VotingClassifier(estimators=[('m_nb', MultinomialNB()),
                                     ('lda', LinearDiscriminantAnalysis()),
                                     ('mlp', MLPClassifier(solver='lbfgs',
                                                           alpha=1e-5,
                                                           hidden_layer_sizes=(10),
                                                           random_state=1))],
                         voting='soft'),
        StackingClassifier(estimators=[('m_nb', MultinomialNB()),
                                     ('lda', LinearDiscriminantAnalysis()),
                                     ('mlp', MLPClassifier(solver='lbfgs',
                                                           alpha=1e-5,
                                                           hidden_layer_sizes=(10),
                                                           random_state=1))],
                           final_estimator=DecisionTreeClassifier())
    ]
    if BINARIZE:
        # If binary classification, we can use unsupervised classifiers also
        cont_perc = cont_perc if cont_perc < 0.5 else 0.5
        learners.extend([COPOD(contamination=cont_perc),
                         IForest(contamination=cont_perc),
                         LOF(n_neighbors=3, contamination=cont_perc)])
    return learners


# ----------------------- MAIN ROUTINE ---------------------


if __name__ == '__main__':

    with open(SCORES_FILE, 'w') as f:
        f.write("dataset_tag,clf,binary,tt_split,acc,mcc,time,model_size\n")

    # Iterating over CSV files in folder
    for dataset_file in os.listdir(CSV_FOLDER):
        full_name = os.path.join(CSV_FOLDER, dataset_file)
        if full_name.endswith(".csv"):
            # if file is a CSV, it is assumed to be a dataset to be processed
            df = pandas.read_csv(full_name, sep=",")
            if VERBOSE:
                print("\n------------ DATASET INFO -----------------")
                print("Data Points in Dataset '%s': %d" % (dataset_file, len(df.index)))
                print("Features in Dataset: " + str(len(df.columns)))

            # Filling NaN and Handling (Removing) constant features
            df = df.fillna(0)
            df = df.loc[:, df.nunique() > 1]
            if VERBOSE:
                print("Features in Dataframe after removing constant ones: " + str(len(df.columns)))

            features_no_cat = df.select_dtypes(exclude=['object']).columns
            if VERBOSE:
                print("Features in Dataframe after non-numeric ones (including label): " + str(len(features_no_cat)))

            # Binarize if needed (for anomaly detection you need a 2-class problem, requires one of the classes to have NORMAL_TAG)
            normal_perc = None
            y = df[LABEL_NAME].to_numpy()
            if BINARIZE:
                y = numpy.where(df[LABEL_NAME] == "normal", 0, 1)
                if VERBOSE:
                    normal_frame = df.loc[df[LABEL_NAME] == NORMAL_TAG]
                    normal_perc = len(normal_frame.index) / len(df.index)
                    print("Normal data: " + str(len(normal_frame.index)) + " items (" +
                          "{:.3f}".format(100.0 * normal_perc) + "%)")
            elif VERBOSE:
                print("Dataset contains %d Classes" % len(numpy.unique(y)))

            # Set up train test split excluding categorical values that some algorithms cannot handle
            # 1-Hot-Encoding or other approaches may be used instead of removing
            x_no_cat = df.select_dtypes(exclude=['object']).to_numpy()
            # Scales the data and saves the scaler
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(x_no_cat)
            dump(scaler, os.path.join( UTILS_FOLDER,"%s_scaler.joblib" % full_name))
            x_train, x_test, y_train, y_test = ms.train_test_split(x_scaled, y, test_size=TT_SPLIT, shuffle=True)

            if VERBOSE:
                print('-------------------- CLASSIFIERS -----------------------')

            # Loop for training and testing each learner specified by LEARNER_TAGS
            for classifier in get_learners(1 - normal_perc):

                # Training the algorithm to get a model
                start_time = current_milli_time()
                classifier.fit(x_train, y_train)

                # Quantifying size of the model
                dump(classifier, "clf_dump.bin")
                size = os.stat("clf_dump.bin").st_size
                os.remove("clf_dump.bin")

                # Saving the model
                dump(classifier,os.path.join(UTILS_FOLDER, "%s_%s.joblib" % (full_name, type(classifier).__name__)))

                # Computing metrics
                y_pred = classifier.predict(x_test)
                acc = metrics.accuracy_score(y_test, y_pred)
                mcc = abs(metrics.matthews_corrcoef(y_test, y_pred))
                if BINARIZE:
                    # Prints metrics for binary classification + train time and model size
                    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
                    print('%s\t-> TP: %d, TN: %d, FP: %d, FN: %d, Accuracy: %.3f, MCC: %.3f - train time: %d ms'
                          ' - model size: %.3f KB' % (classifier.__class__.__name__, tp, tn, fp, fn, acc, mcc,
                                                      current_milli_time() - start_time, size / 1000.0))
                else:
                    # Prints just accuracy for multi-class classification problems, no confusion matrix
                    print('%s\t-> Accuracy: %.3f, MCC: %.3f - train time: %d ms - model size: %.3f KB'
                          % (classifier.__class__.__name__, acc, current_milli_time() - start_time, size / 1000.0))

                # Updates CSV file form metrics of experiment
                with open(SCORES_FILE, "a") as myfile:
                    # Prints result of experiment in CSV file
                    myfile.write(full_name + "," + classifier.__class__.__name__ + "," + str(BINARIZE) + "," +
                                 str(TT_SPLIT) + ',' + str(acc) + "," + str(mcc) + "," +
                                 str(current_milli_time() - start_time) + "," + str(size) + "\n")