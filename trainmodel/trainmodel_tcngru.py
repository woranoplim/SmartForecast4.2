#
# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# from torch.utils.data import TensorDataset, DataLoader
# from datetime import timedelta
#
# # ---------- Load Data ----------
# ticker = "NVDA"
# # df = pd.read_csv(f"TESTWEB/datasets/{ticker}_dataset.csv", index_col=0, parse_dates=True)
# data = pd.read_csv(f"{ticker}_datasettest.csv", index_col=0, parse_dates=True)
#
# features = ['Close', 'High', 'Low', 'Open', 'Volume',"VIX_Close",
#             'EMA_12', 'EMA_26', 'RSI', 'MACD',
#             'MA_10', 'MA_20', 'MA_30', 'MA_45',
#             'SD_20', 'Upper_20', 'Lower_20']
# df = df[features].dropna().copy()
# close_prices = df["Close"].astype(float).values
# scaler_x = MinMaxScaler()
# scaled_features = scaler_x.fit_transform(df[features])
#
# # ---------- Target = %change ----------
# forecast_horizon = 30
# step = 1
# seq_len = 300
#
# X, y = [], []
# for i in range(len(scaled_features) - seq_len - forecast_horizon * step + 1):
#     X.append(scaled_features[i:i + seq_len])
#     base = close_prices[i + seq_len - 1]
#     target_prices = [
#         close_prices[i + seq_len + j * step - 1] for j in range(1, forecast_horizon + 1)
#     ]
#     pct_change = (np.array(target_prices) - base) / base
#     y.append(pct_change)
#
# X, y = np.array(X), np.array(y)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
#
# # ---------- Hybrid TCN + GRU Model ----------
# class TCNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation):
#         super().__init__()
#         padding = (kernel_size - 1) * dilation // 2
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
#                               padding=padding, dilation=dilation)
#         self.norm = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         return self.relu(self.norm(self.conv(x)))
#
# class HybridTCN_GRU(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super().__init__()
#         self.tcn = nn.Sequential(
#             TCNBlock(num_inputs, 64, kernel_size=3, dilation=1),
#             nn.Dropout(0.2),
#             TCNBlock(64, 64, kernel_size=3, dilation=2),
#             nn.Dropout(0.2),
#             TCNBlock(64, 64, kernel_size=3, dilation=4),
#             nn.Dropout(0.2),
#         )
#         self.gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
#         self.fc = nn.Linear(64, num_outputs)
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # [B, C, T]
#         x = self.tcn(x)         # [B, 64, T]
#         x = x.permute(0, 2, 1)  # [B, T, 64]
#         _, h = self.gru(x)      # h: [1, B, 64]
#         return self.fc(h.squeeze(0))
#
# # ---------- Train ----------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# model = HybridTCN_GRU(num_inputs=X.shape[2], num_outputs=forecast_horizon).to(device)
# loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
#
# train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
#                               torch.tensor(y_train, dtype=torch.float32))
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
# y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
#
# best_val_loss = float('inf')
# patience = 5
# counter = 0
#
# for epoch in range(100):
#     model.train()
#     total_loss = 0
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         optimizer.zero_grad()
#         output = model(xb)
#         loss = loss_fn(output, yb)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#
#     model.eval()
#     with torch.no_grad():
#         val_output = model(X_val_tensor)
#         val_loss = loss_fn(val_output, y_val_tensor).item()
#
#     print(f"Epoch {epoch} - Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
#     scheduler.step(val_loss)
#
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), f"{ticker}_tcn_gru.pt")
#         counter = 0
#     else:
#         counter += 1
#         if counter >= patience:
#             print("Early stopping.")
#             break
#
# # ---------- Evaluate ----------
# model.load_state_dict(torch.load(f"{ticker}_tcn_gru.pt"))
# model.eval()
# y_pred = model(X_val_tensor).detach().cpu().numpy()
#
# true_prices, pred_prices = [], []
# for i in range(len(y_val)):
#     base = close_prices[len(X_train) + seq_len + i - 1]
#     true_seq = base * (1 + y_val[i])
#     pred_seq = base * (1 + y_pred[i])
#     true_prices.extend(true_seq)
#     pred_prices.extend(pred_seq)
#
# mae = mean_absolute_error(true_prices, pred_prices)
# mape = np.mean(np.abs((np.array(true_prices) - np.array(pred_prices)) / np.array(true_prices))) * 100
# r2 = r2_score(true_prices, pred_prices)
# print(f"\n📊 Hybrid TCN+GRU MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
#
# # ---------- Predict 30-day Ahead ----------
# last_seq = scaled_features[-seq_len:]
# last_close = close_prices[-1]
# last_seq_tensor = torch.tensor(last_seq.reshape(1, seq_len, X.shape[2]), dtype=torch.float32).to(device)
# future_pct = model(last_seq_tensor).detach().cpu().numpy()[0]
# future_prices = last_close * (1 + future_pct)
#
# # ---------- Plot ----------
# lookback_days = 60
# actual_dates = df.index[-lookback_days:]
# actual_prices = close_prices[-lookback_days:]
# last_date = df.index[-1]
# future_dates = [last_date + timedelta(days=(i + 1) * step) for i in range(forecast_horizon)]
#
# plt.figure(figsize=(14, 6))
# plt.plot(actual_dates, actual_prices, label="Actual Prices", color='green')
# plt.plot(future_dates, future_prices, label="Hybrid TCN+GRU Prediction", marker='o', color='orange')
# plt.axvline(x=last_date, color='gray', linestyle='--', label='Prediction Start')
# plt.title(f"{ticker} Forecast: Hybrid TCN+GRU (Next {forecast_horizon * step} Days)")
# plt.xlabel("Date")
# plt.ylabel("Price (USD)")
# plt.grid(True)
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------- Load Data ----------
ticker = "META"
df = pd.read_csv(f"{ticker}_datasettest.csv", index_col=0, parse_dates=True)

features = ['Close', 'High', 'Low', 'Open', 'Volume', "VIX_Close",
            'EMA_12', 'EMA_26', 'RSI', 'MACD',
            'MA_10', 'MA_20', 'MA_30', 'MA_45',
            'SD_20', 'Upper_20', 'Lower_20']
df = df[features].dropna().copy()
close_prices = df["Close"].astype(float).values

scaler_x = MinMaxScaler()
scaled_features_all = scaler_x.fit_transform(df[features])

# ---------- Target = %change setup ----------
forecast_horizon = 30
step = 1

def make_sequences(scaled_features, close_prices, seq_len, horizon, step):
    X, y = [], []
    max_i = len(scaled_features) - seq_len - horizon * step + 1
    if max_i <= 0:
        return np.empty((0, seq_len, scaled_features.shape[1])), np.empty((0, horizon))
    for i in range(max_i):
        X.append(scaled_features[i:i + seq_len])
        base = close_prices[i + seq_len - 1]
        target_prices = [close_prices[i + seq_len + j * step - 1] for j in range(1, horizon + 1)]
        pct_change = (np.array(target_prices) - base) / base
        y.append(pct_change)
    return np.array(X), np.array(y)

# ---------- Hybrid TCN + GRU Model ----------
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class HybridTCN_GRU(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(num_inputs, 64, kernel_size=3, dilation=1),
            nn.Dropout(0.2),
            TCNBlock(64, 64, kernel_size=3, dilation=2),
            nn.Dropout(0.2),
            TCNBlock(64, 64, kernel_size=3, dilation=4),
            nn.Dropout(0.2),
        )
        self.gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_outputs)
    def forward(self, x):
        x = x.permute(0, 2, 1)   # [B, C, T]
        x = self.tcn(x)          # [B, 64, T]
        x = x.permute(0, 2, 1)   # [B, T, 64]
        _, h = self.gru(x)       # h: [1, B, 64]
        return self.fc(h.squeeze(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Sweep over seq_len ----------
seq_list = [90, 120, 150, 180, 240, 300]
results = []
best_rec = None  # keep (val_loss, seq_len, state_dict, scaler_x) etc.

for seq_len in seq_list:
    # Build sequence data
    X, y = make_sequences(scaled_features_all, close_prices, seq_len, forecast_horizon, step)
    if len(X) == 0:
        print(f"⚠️ ข้าม seq_len={seq_len} (ข้อมูลไม่พอ)")
        continue

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Datasets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Model / Optim / Sched
    model = HybridTCN_GRU(num_inputs=X.shape[2], num_outputs=forecast_horizon).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)

    # Train
    best_val = float('inf')
    patience = 5
    bad = 0
    for epoch in range(100):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            vout = model(X_val_tensor)
            vloss = loss_fn(vout, y_val_tensor).item()
        scheduler.step(vloss)
        # print(f"[seq={seq_len}] Epoch {epoch} - Train {total/len(train_loader):.4f}  Val {vloss:.4f}")

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                # print(f"[seq={seq_len}] Early stopping at epoch {epoch}")
                break

    # Evaluate on validation (price space)
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        y_pred = model(X_val_tensor).detach().cpu().numpy()

    true_prices, pred_prices = [], []
    # index mapping to close_prices
    # train_size = len(X_train)  (train split in sequence space)
    train_size = len(X_train)
    for i in range(len(y_val)):
        base = close_prices[train_size + seq_len + i - 1]
        true_seq = base * (1 + y_val[i])
        pred_seq = base * (1 + y_pred[i])
        true_prices.extend(true_seq)
        pred_prices.extend(pred_seq)

    mae = mean_absolute_error(true_prices, pred_prices)
    mape = np.mean(np.abs((np.array(true_prices) - np.array(pred_prices)) / np.array(true_prices))) * 100
    r2 = r2_score(true_prices, pred_prices)
    results.append({"seq_len": seq_len, "MAE": mae, "MAPE": mape, "R2": r2, "val_loss": best_val})
    print(f"[seq_len={seq_len}]  MAE: {mae:.4f}  MAPE: {mape:.2f}%  R²: {r2:.4f}  (best val_loss: {best_val:.5f})")

    # keep best by lowest MAE (หรือจะใช้ val_loss ก็ได้)
    tie_break = 1e-12
    if (best_rec is None) or (mae < best_rec["MAE"] - tie_break):
        best_rec = {
            "seq_len": seq_len,
            "MAE": mae, "MAPE": mape, "R2": r2, "val_loss": best_val,
            "state": best_state
        }

# ---------- Summary ----------
summary_df = pd.DataFrame(results).sort_values("seq_len")
print("\n==== Summary over seq_len ====")
print(summary_df.to_string(index=False))

# ---------- Use best seq_len to predict next 30 days ----------
if best_rec is None:
    raise RuntimeError("No valid seq_len produced data. Check your dataset length.")

best_seq = best_rec["seq_len"]
print(f"\n✅ Best seq_len = {best_seq}  (MAE={best_rec['MAE']:.4f}, MAPE={best_rec['MAPE']:.2f}%, R²={best_rec['R2']:.4f})")

# เซฟโมเดลที่ดีที่สุดเป็น .pt
best_model_path = f"tcn_gru_{ticker}_{best_seq}.pt"
torch.save(best_rec["state"], best_model_path)
print(f"💾 Saved best model weights to {best_model_path}")

# Rebuild X,y for best seq_len
X_best, y_best = make_sequences(scaled_features_all, close_prices, best_seq, forecast_horizon, step)
train_size_best = int(len(X_best) * 0.8)

# Build & load best weights
model_best = HybridTCN_GRU(num_inputs=X_best.shape[2], num_outputs=forecast_horizon).to(device)
model_best.load_state_dict(torch.load(best_model_path, map_location=device))
model_best.eval()


# ---------- Use best seq_len to predict next 30 days ----------
if best_rec is None:
    raise RuntimeError("No valid seq_len produced data. Check your dataset length.")

# Rebuild X,y for best seq_len
X_best, y_best = make_sequences(scaled_features_all, close_prices, best_seq, forecast_horizon, step)
# Train/val split (for indexing base in plot)
train_size_best = int(len(X_best) * 0.8)
X_train_b, X_val_b = X_best[:train_size_best], X_best[train_size_best:]
y_train_b, y_val_b = y_best[:train_size_best], y_best[train_size_best:]

# Build & load best weights
model_best = HybridTCN_GRU(num_inputs=X_best.shape[2], num_outputs=forecast_horizon).to(device)
model_best.load_state_dict(best_rec["state"])
model_best.eval()

# Predict 30-day ahead from last window
last_seq = scaled_features_all[-best_seq:]
last_close = close_prices[-1]
last_seq_tensor = torch.tensor(last_seq.reshape(1, best_seq, X_best.shape[2]), dtype=torch.float32).to(device)
with torch.no_grad():
    future_pct = model_best(last_seq_tensor).detach().cpu().numpy()[0]
future_prices = last_close * (1 + future_pct)

# ---------- Plot ----------
lookback_days = 60
actual_dates  = df.index[-lookback_days:]
actual_prices = close_prices[-lookback_days:]
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=(i + 1) * step) for i in range(forecast_horizon)]

plt.figure(figsize=(14, 6))
plt.plot(actual_dates, actual_prices, label="Actual Prices", color='green')
plt.plot(future_dates, future_prices, label=f"Hybrid TCN+GRU (seq_len={best_seq})", marker='o', color='orange')
plt.axvline(x=last_date, color='gray', linestyle='--', label='Prediction Start')
plt.title(f"{ticker} Forecast: Hybrid TCN+GRU (Next {forecast_horizon * step} Days)")
plt.xlabel("Date"); plt.ylabel("Price (USD)")
plt.grid(True); plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.show()

