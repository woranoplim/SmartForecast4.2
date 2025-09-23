import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import json
import os

# ---------- Parameters ----------
ticker = "META"
forecast_horizon = 30
step = 1
seq_length_lstm = 90
seq_length_trans = 90
seq_length_tcn = 90

# ---------- Learnable Positional Encoding แบบใหม่ ----------
class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):  # ← รองรับ kwargs จากการโหลดโมเดล
        super().__init__(**kwargs)  # ← ส่งต่อไปยัง Layer base class
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=sequence_length, output_dim=d_model)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_encoding = self.pos_embedding(positions)
        return x + pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
        })
        return config


# ---------- GRU + TCN ----------
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
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
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

# ---------- Sequence Generator ----------
def create_sequences(features, close_prices, seq_len, horizon, step):
    X, y = [], []
    for i in range(len(features) - seq_len - horizon * step + 1):
        X.append(features[i:i+seq_len])
        targets = []
        base_price = close_prices[i + seq_len - 1]
        for j in range(1, horizon + 1):
            future_price = close_prices[i + seq_len + j*step - 1]
            pct_change = (future_price - base_price) / base_price
            targets.append(pct_change)
        y.append(targets)
    return np.array(X), np.array(y)

# ---------- Load and prepare data ----------
data = pd.read_csv(f"datasets/{ticker}_dataset.csv", index_col=0, parse_dates=True)
features = ['Close', 'High', 'Low', 'Open', 'Volume',"VIX_Close",
            'EMA_12', 'EMA_26', 'RSI', 'MACD',
            'MA_10', 'MA_20', 'MA_30', 'MA_45',
            'SD_20', 'Upper_20', 'Lower_20']
data = data[features].dropna()
close_prices = data["Close"].values.astype(float)
scaled_features = MinMaxScaler().fit_transform(data[features])

X_lstm, _ = create_sequences(scaled_features, close_prices, seq_length_lstm, forecast_horizon, step)
X_trans, _ = create_sequences(scaled_features, close_prices, seq_length_trans, forecast_horizon, step)
X_tcn, _ = create_sequences(scaled_features, close_prices, seq_length_tcn, forecast_horizon, step)

last_seq_lstm = X_lstm[-1]
last_seq_trans = X_trans[-1]
last_seq_tcn = X_tcn[-1]
last_close = close_prices[-1]
last_date = data.index[-1]
future_dates = [last_date + timedelta(days=(i + 1) * step) for i in range(forecast_horizon)]

# ---------- Load models ----------
lstm_model = load_model("model/lstm_META.h5", compile=False)
transformer_model = load_model(
    "model/transformer_META.h5",
    compile=False,
    custom_objects={"LearnablePositionalEncoding": LearnablePositionalEncoding}
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tcn_gru_model = HybridTCN_GRU(num_inputs=X_tcn.shape[2], num_outputs=forecast_horizon).to(device)
try:
    tcn_gru_model.load_state_dict(torch.load(f"model/tcn_gru_{ticker}_{seq_length_tcn}.pt", map_location=device))
    tcn_gru_model.eval()
    tcn_available = True
except FileNotFoundError:
    print("⚠️ ไม่พบไฟล์ TCN-GRU สำหรับ META")
    tcn_available = False

# ---------- Predict ----------
lstm_pred_pct = lstm_model.predict(last_seq_lstm.reshape(1, seq_length_lstm, X_lstm.shape[2]))[0]
trans_pred_pct = transformer_model.predict(last_seq_trans.reshape(1, seq_length_trans, X_trans.shape[2]))[0]
input_tensor = torch.tensor(last_seq_tcn.reshape(1, seq_length_tcn, X_tcn.shape[2]), dtype=torch.float32).to(device)
tcn_pred_pct = tcn_gru_model(input_tensor).detach().cpu().numpy()[0]

lstm_future_prices = last_close * (1 + lstm_pred_pct)
trans_future_prices = last_close * (1 + trans_pred_pct)
tcn_future_prices = last_close * (1 + tcn_pred_pct)

# ---------- % Return ----------
lstm_return_pct = ((lstm_future_prices[-1] - last_close) / last_close) * 100
trans_return_pct = ((trans_future_prices[-1] - last_close) / last_close) * 100
tcn_return_pct = ((tcn_future_prices[-1] - last_close) / last_close) * 100

# ---------- Print results ----------
lstm_return_pct = ((lstm_future_prices[-1] - last_close) / last_close) * 100
trans_return_pct = ((trans_future_prices[-1] - last_close) / last_close) * 100
print(f"LSTM คาดว่าอีก {forecast_horizon * step} วันจะ {'เพิ่มขึ้น📈' if lstm_return_pct >= 0 else 'ลดลง📉'} {lstm_return_pct:.2f}%")
print(f"Transformer คาดว่าอีก {forecast_horizon * step} วันจะ {'เพิ่มขึ้น📈' if trans_return_pct >= 0 else 'ลดลง📉'} {trans_return_pct:.2f}%")
print(f"Hybrid TCN+GRU คาดว่าอีก {forecast_horizon * step} วันจะ {'เพิ่มขึ้น📈' if tcn_return_pct >= 0 else 'ลดลง📉'} {tcn_return_pct:.2f}%")
print("\n🔎 LSTM Model Summary")
lstm_model.summary()

print("\n🔎 Transformer Model Summary")
transformer_model.summary()
# ---------- Save JSON ----------
results_df = pd.DataFrame({
    "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
    "LSTM_Predicted_Price": lstm_future_prices,
    "Transformer_Predicted_Price": trans_future_prices,
    "TCN_GRU_Predicted_Price": tcn_future_prices,
})
results_df.loc[-1] = {
    "Date": last_date.strftime("%Y-%m-%d"),
    "LSTM_Predicted_Price": last_close,
    "Transformer_Predicted_Price": last_close,
    "TCN_GRU_Predicted_Price": last_close,
}
results_df.sort_index(inplace=True)

os.makedirs("json", exist_ok=True)
with open(f"json/{ticker}.json", "w") as f:
    json.dump(results_df.to_dict(orient="records"), f, indent=4)

print(f"✅ บันทึกผลลัพธ์ {ticker}.json แล้ว")
