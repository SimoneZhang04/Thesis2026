import os

import pandas
import sklearn.model_selection as ms

# Sub folder where the .csv file will be taken from
SOURCE_SUB_FOLDER = 'HW_Failure'

SOURCE_FOLDER = os.path.join('difficulty', SOURCE_SUB_FOLDER)
OUTPUT_FOLDER = 'calculate_difficullty'
TEST_FOLDER = 'test'
TRAIN_FOLDER = 'train'
MULTILABEL_COLUMN = 'multilabel'
LABEL_COLUMN = 'label'
SPLIT_SIZE = 0.3

if __name__ == '__main__':

    # Iterating over .csv files in folder
    for dataset_file in os.listdir(SOURCE_FOLDER):
        full_name = os.path.join(SOURCE_FOLDER, dataset_file)
        if full_name.endswith(".csv"):
            df = pandas.read_csv(full_name, sep=",")

            # if there's a multilabel then split with it as stratify value, else with label
            if MULTILABEL_COLUMN in df.columns:
                stratify_column = MULTILABEL_COLUMN
            else:
                   stratify_column = LABEL_COLUMN

            df_train, df_test = ms.train_test_split(df, test_size=SPLIT_SIZE, stratify=df[stratify_column], random_state=42)

            df_train.to_csv(os.path.join(OUTPUT_FOLDER, TRAIN_FOLDER, SOURCE_SUB_FOLDER, dataset_file), index=False)
            df_test.to_csv(os.path.join(OUTPUT_FOLDER, TEST_FOLDER, SOURCE_SUB_FOLDER, dataset_file), index=False)


