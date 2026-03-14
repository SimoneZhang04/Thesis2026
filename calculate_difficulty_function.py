import os

import joblib
import numpy
import pandas

import main_tabulardata

# SCALER SAVE FILE  "%s_scaler.joblib" % full_name
# CLASSIFIER SAVE FILE "%s_%s.joblib" % (full_name, type(classifier).__name__)
if __name__ == '__main__':

    for dataset_file in os.listdir(main_tabulardata.CSV_FOLDER):
        full_name = os.path.join(main_tabulardata.CSV_FOLDER, dataset_file)
        if full_name.endswith(".csv"):
            # if file is a CSV, it is assumed to be a dataset
            df_original = pandas.read_csv(full_name, sep=",")
            # Filling NaN and Handling (Removing) constant features
            df = df_original.fillna(0)
            df = df.loc[:, df.nunique() > 1]

            # Binarize if needed (for anomaly detection you need a 2-class problem, requires one of the classes to have NORMAL_TAG)
            normal_perc = None
            y = df[main_tabulardata.LABEL_NAME].to_numpy()
            if main_tabulardata.BINARIZE:
                y = numpy.where(df[main_tabulardata.LABEL_NAME] == "normal", 0, 1)
                if main_tabulardata.VERBOSE:
                    normal_frame = df.loc[df[main_tabulardata.LABEL_NAME] == main_tabulardata.NORMAL_TAG]
                    normal_perc = len(normal_frame.index) / len(df.index)
                    print("Normal data: " + str(len(normal_frame.index)) + " items (" +
                          "{:.3f}".format(100.0 * normal_perc) + "%)")
            elif main_tabulardata.VERBOSE:
                print("Dataset contains %d Classes" % len(numpy.unique(y)))

            x_no_cat = df.select_dtypes(exclude=['object']).to_numpy()
            scaler = joblib.load("%s_scaler.joblib" % full_name)
            x_scaled = scaler.fit_transform(x_no_cat)

            # Calculating the score function
            score_col =  []
            for classifier in main_tabulardata.get_learners(1 - normal_perc):
                model = joblib.load("%s_%s.joblib" % (full_name, type(classifier).__name__))

                scores = abs(model.predict(x_scaled) - y)
                # Adds the score function value of the classifier to each input value (0 for correct, 1 wrong prediction)
                df_original["%s_score" % type(classifier).__name__] = scores
                score_col.append("%s_score" % type(classifier).__name__)

            # Calculates the difficulty function by getting a mean of each score
            df_original['difficulty_function'] = df_original.filter(score_col).mean(axis=1)
            # Saves the file to .csv with limit of 9 decimals (for difficulty function)
            df_original.to_csv(os.path.join(" difficulty", "%s_with_difficulty_function.csv" % full_name), index=False, float_format = '%.9f')
