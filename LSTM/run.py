# main.py

import os
import json
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from core.data_processor import DataLoader
from core.LSTM_modified_model import Model
import warnings
warnings.filterwarnings("ignore")

current_time = datetime.now().strftime("%Y%m%d%H")

def output_results_and_errors_multiple(predicted_data, true_data, true_data_base, prediction_len, file_name,
                                       sentiment_type, num_csvs):
    save_df = pd.DataFrame()
    save_df['True_Data'] = true_data.reshape(-1)
    save_df['Base'] = true_data_base.reshape(-1)
    save_df['True_Data_origin'] = (save_df['True_Data'] + 1) * save_df['Base']

    if predicted_data:
        all_predicted_data = np.concatenate([p for p in predicted_data])
    else:
        all_predicted_data = predicted_data

    save_df['Predicted_Data'] = pd.Series(all_predicted_data)
    save_df['Predicted_Data_origin'] = (save_df['Predicted_Data'] + 1) * save_df['Base']
    save_df = save_df.fillna(np.nan)

    result_folder = "test_results"
    os.makedirs(result_folder, exist_ok=True)

    save_file_path = os.path.join(result_folder, f"{file_name}_predict.csv")
    save_df.to_csv(save_file_path, index=False)
    print(f"Data saved to {save_file_path}")

    min_length = min(len(save_df['Predicted_Data']), len(save_df['True_Data']))
    predicted_data = save_df['Predicted_Data'][:min_length]
    true_data = save_df['True_Data'][:min_length]

    mae = mean_absolute_error(true_data, predicted_data)
    mse = mean_squared_error(true_data, predicted_data)
    r2 = r2_score(true_data, predicted_data)

    results_df = pd.DataFrame({
        'MAE': [mae],
        'MSE': [mse],
        'R2': [r2]
    })

    eval_file_path = os.path.join(result_folder, f"{file_name}_eval.csv")
    results_df.to_csv(eval_file_path, index=False)
    print(f"\nResults saved to {eval_file_path}")

def main(configs, data_filename, sentiment_type, flag_pred, model_name):
    symbol_name = data_filename.split('.')[0]
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', data_filename),
        configs['data']['train_test_split'],
        configs['data']['columns'],
        configs['data']['columns_to_normalise'],
        configs['data']['prediction_length']
    )

    model = Model()
    model_path = f"saved_models/{model_name}_{sentiment_type}_{symbol_name}.h5"
    
    if os.path.exists(model_path):
        model.load_model(model_path, configs)
    else:
        model.build_model(configs)
        x, y = data.get_train_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
        steps_per_epoch = math.ceil(
            (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size']
        )
        model.train_generator(
            data_gen=data.generate_train_batch(
                seq_len=configs['data']['sequence_length'],
                batch_size=configs['training']['batch_size'],
                normalise=configs['data']['normalise']
            ),
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            steps_per_epoch=steps_per_epoch,
            save_dir=configs['model']['save_dir'],
            sentiment_type=sentiment_type,
            model_name=model_name,
            data_filename=symbol_name
        )

    if flag_pred:
        x_test, y_test, y_base = data.get_test_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise'],
            cols_to_norm=configs['data']['columns_to_normalise']
        )
        print("Test data shape:")
        print("X:", x_test.shape)
        print("Y:", y_test.shape)
        predictions = model.predict_sequences_multiple_modified(
            x_test,
            configs['data']['sequence_length'],
            configs['data']['prediction_length']
        )
        print("outputing")
        output_results_and_errors_multiple(predictions, y_test, y_base,
                                           configs['data']['prediction_length'],
                                           symbol_name, sentiment_type, data_filename)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    data_dir = 'data'  # đường dẫn tới thư mục chứa các file .csv
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    file_names = [os.path.splitext(f)[0] for f in csv_files]
    model_name = "LSTM"
    sentiment_types = ["sentiment"] 
    all_names = csv_files

    for name in all_names:
        for sentiment_type in sentiment_types:
            print(name)
            configs = json.load(open(sentiment_type + '_config.json', 'r'))
            main(configs, name, sentiment_type, True, model_name)