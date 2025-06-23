import os
import json
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.Autoformer import Model as Autoformer
from core.autoformer_dataloader import AutoformerDataLoader

def dict2namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict2namespace(v)
    return Namespace(**d)

def output_results_and_errors(predicted, true, scaler, file_name):
    pred_flat = predicted.reshape(-1, predicted.shape[-1])
    true_flat = true.reshape(-1, true.shape[-1])

    # 2) Dua ve thang goc
    pred_orig = scaler.inverse_transform(pred_flat)
    true_orig = scaler.inverse_transform(true_flat)

    y_pred = pred_orig[:, 0]
    y_true = true_orig[:, 0]

    # 3) Metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)

    # 4) Luu
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)

    # 5) Luu file pred vs true
    df = pd.DataFrame({
        "Predicted": y_pred,
        "True":      y_true
    })
    pred_path = os.path.join(output_dir, f"{file_name}_predict.csv")
    df.to_csv(pred_path, index=False)

    # 6) Luu file metrics
    metrics_df = pd.DataFrame([{"MAE": mae, "MSE": mse, "R2": r2}])
    eval_path = os.path.join(output_dir, f"{file_name}_eval.csv")
    metrics_df.to_csv(eval_path, index=False)

    print(f"[INFO] Saved {pred_path} & {eval_path}")

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for i in range(loader[0].shape[0]):
        x_enc = torch.tensor(loader[0][i:i+1], dtype=torch.float32).to(device)
        x_mark_enc = torch.tensor(loader[1][i:i+1], dtype=torch.float32).to(device)
        x_dec = torch.tensor(loader[2][i:i+1], dtype=torch.float32).to(device)
        x_mark_dec = torch.tensor(loader[3][i:i+1], dtype=torch.float32).to(device)
        y = torch.tensor(loader[4][i:i+1], dtype=torch.float32).to(device)
        optimizer.zero_grad()
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / loader[0].shape[0]

def predict_all(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(loader[0].shape[0]):
            x_enc = torch.tensor(loader[0][i:i+1], dtype=torch.float32).to(device)
            x_mark_enc = torch.tensor(loader[1][i:i+1], dtype=torch.float32).to(device)
            x_dec = torch.tensor(loader[2][i:i+1], dtype=torch.float32).to(device)
            x_mark_dec = torch.tensor(loader[3][i:i+1], dtype=torch.float32).to(device)
            out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)

def main(config_path, csv_path, model_save_path="saved_models/autoformer.pth", predict_only=False):
    with open(config_path, 'r') as f:
        config = json.load(f)
        configs = dict2namespace(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("test_results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    # Load data
    data = AutoformerDataLoader(csv_path, configs)
    train_data = data.get_train()
    test_data = data.get_test()

    # Model
    model = Autoformer(configs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    if not predict_only:
        print("[INFO] Training...")
        for epoch in range(configs.epochs):
            loss = train_one_epoch(model, train_data, optimizer, criterion, device)
            print(f"Epoch {epoch+1}/{configs.epochs} - Loss: {loss:.6f}")
            torch.save(model.state_dict(), model_save_path)

    print("[INFO] Predicting on test set...")
    model.load_state_dict(torch.load(model_save_path))
    preds = predict_all(model, test_data, device)
    output_results_and_errors(preds, test_data[4], data.scaler, os.path.splitext(os.path.basename(csv_path))[0])

if __name__ == "__main__":
    main("sentiment_config.json", "data/DIS.csv")
