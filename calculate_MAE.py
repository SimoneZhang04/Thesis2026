import os

import pandas
from sklearn.metrics import mean_absolute_error


# Folders to get and save the data
# OPTIONAL VARS
SOURCE_FOLDER = 'calculate_difficulty'
SUB_FOLDER = 'HW_Failure'
OUTPUT_FOLDER = 'MAE'
FILE_NAME = 'BackBlaze_2017_5PercRate_scikit.csv'
# NEEDED VARS
DIFFICULTY_COLUMN = 'difficulty_function'
PREDICTED_DIFFICULTY_COLUMN = 'predicted_difficulty_function'
CONFIDENCE_COLUMN = 'confidence'
TIME_COLUMN = 'time'
FULL_FILE_NAME = os.path.join(SOURCE_FOLDER, 'test', SUB_FOLDER, FILE_NAME)
FULL_OUTPUT_NAME = os.path.join(OUTPUT_FOLDER, SUB_FOLDER, FILE_NAME)

def calculate_mae(dataframe):
    results = []
    dif_columns = dataframe.filter(like=PREDICTED_DIFFICULTY_COLUMN).columns.tolist()
    conf_columns = dataframe.filter(like=CONFIDENCE_COLUMN).columns.tolist()
    dataframe_filtered = dataframe[dif_columns + [DIFFICULTY_COLUMN] + conf_columns].dropna()

    pred_columns = dataframe_filtered[dif_columns]

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

        ae_with_confidence = dataframe[confidence_column] * abs(y_pred - y_true)
        ae_with_confidence_mean = ae_with_confidence.mean()

        time_for_row = dataframe[TIME_COLUMN + '_with_' + suffix]
        results.append({
            'Algorithm': prediction_column,
            'MAE': round(mae, 6),
            'Mean_of_confidence': round(confidence_mean, 6),
            'MAE_with_confidence >= 0.99' : mae_with_conf(confidence_column, dataframe_filtered, prediction_column, 0.99),
            'MAE_with_confidence >= 0.95': mae_with_conf(confidence_column, dataframe_filtered, prediction_column, 0.95),
            'MAE_with_confidence >= 0.9': mae_with_conf(confidence_column, dataframe_filtered, prediction_column, 0.9),
            'MAE_with_confidence >= 0.8': mae_with_conf(confidence_column, dataframe_filtered, prediction_column, 0.8),
            'MAE_with_confidence >= 0.7': mae_with_conf(confidence_column, dataframe_filtered, prediction_column, 0.7),
            'Error_multiplied_confidence ' : ae_with_confidence_mean,
            'Number_of_data_without_confidence >=0.99 ' : count_conf(confidence_column, dataframe_filtered, 0.99),
            'Number_of_data_without_confidence >=0.95 ' : count_conf(confidence_column, dataframe_filtered, 0.95),
            'Number_of_data_without_confidence >=0.9 ': count_conf(confidence_column, dataframe_filtered, 0.9),
            'Number_of_data_without_confidence >=0.8 ': count_conf(confidence_column, dataframe_filtered, 0.8),
            'Number_of_data_without_confidence >=0.7 ': count_conf(confidence_column, dataframe_filtered, 0.7),
            # Percentage of data used in the calculations some row are dropped by being nan
            'Total_number_of_data': dataframe.shape[0],
            'time_for_row' : time_for_row [0]
        })

    report_df = pandas.DataFrame(results).sort_values(by='MAE')

    return report_df

# Counts the number of inputs with confidence >= confidence_value
def count_conf(confidence_column, dataframe_filtered, confidence_value):
    dataframe_temp = dataframe_filtered[dataframe_filtered[confidence_column] >= confidence_value]
    try:
        count = dataframe_temp[confidence_column].count()
    except ValueError:
        count = 0

    return  count
# Calculated the mae of value with confidence >= confidence_value
def mae_with_conf(confidence_column, dataframe_filtered, prediction_column, confidence_value):
    dataframe_temp = dataframe_filtered[dataframe_filtered[confidence_column] >= confidence_value]
    y_pred_conf = dataframe_temp[prediction_column]
    y_true_conf = dataframe_temp[DIFFICULTY_COLUMN]
    try:
        mae_conf = mean_absolute_error(y_true_conf, y_pred_conf)
    except ValueError:
        mae_conf = 'No_value'

    return  mae_conf


if __name__ == '__main__':
    df = pandas.read_csv(FULL_FILE_NAME)
    final_df = calculate_mae(df)
    final_df.to_csv(FULL_OUTPUT_NAME, index=False)
