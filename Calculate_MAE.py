import os

import pandas
from sklearn.metrics import mean_absolute_error

# Folders to get and save the data
# OPTIONAL VARS
SOURCE_FOLDER = 'calculate_difficulty'
SUB_FOLDER = 'HW_Failure'
OUTPUT_FOLDER = 'MAE'

# NEEDED VARS
DIFFICULTY_COLUMN = 'difficulty_function'
PREDICTED_DIFFICULTY_COLUMN = 'predicted_difficulty_function'
DISTANCE_COLUMN = 'distance'
CONFIDENCE_COLUMN = 'confidence'
FILE_NAME = 'Baidu_SMART Dataset_15Perc_scikit.csv'
FULL_FILE_NAME = os.path.join(SOURCE_FOLDER, 'test', SUB_FOLDER, FILE_NAME)
FULL_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, SUB_FOLDER, FILE_NAME)

def calculate_mae(dataframe):
    results = []
    columns = dataframe.filter(like=PREDICTED_DIFFICULTY_COLUMN).columns.tolist()

    dataframe_filtered = dataframe[columns + [DIFFICULTY_COLUMN]].dropna()

    pred_columns = dataframe_filtered[columns]

    y_true = dataframe_filtered[DIFFICULTY_COLUMN]
    for prediction_column in pred_columns:

        y_pred = dataframe_filtered[prediction_column]
        # Gets the suffix of the prediction column
        suffix = prediction_column.split('_with_')[-1]
        confidence_column = CONFIDENCE_COLUMN + '_with_' + suffix

        # Calculates MAE
        mae = mean_absolute_error(y_true, y_pred)

        # Calculates mean of the confidence
        confidence_mean = dataframe[confidence_column].mean()
        results.append({
            'Algorithm': prediction_column,
            'MAE': round(mae, 6),
            'Mean_of_confidence': round(confidence_mean, 6),
            # Percentage of data used in the calculations some row are dropped by being nan
            'Percentage_of_data': y_pred.shape[0]/dataframe.shape[0],
        })

    report_df = pandas.DataFrame(results).sort_values(by='MAE')

    return report_df

if __name__ == '__main__':
    df = pandas.read_csv(FULL_FILE_NAME)
    final_df = calculate_mae(df)
    final_df.to_csv(FULL_OUTPUT_FOLDER, index=False)
