import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, LayerNormalization, MultiHeadAttention, Add, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta
import matplotlib.pyplot as plt

# ---------- โหลดและเตรียมข้อมูล ----------
ticker = "MSFT"
data = pd.read_csv(f"{ticker}_datasettest.csv", index_col=0, parse_dates=True)
features = ['Close', 'High', 'Low', 'Open', 'Volume',"VIX_Close",
            'EMA_12', 'EMA_26', 'RSI', 'MACD',
            'MA_10', 'MA_20', 'MA_30', 'MA_45',
            'SD_20', 'Upper_20', 'Lower_20']


data = data[features].dropna()
close_prices = data["Close"].values
print("📅 Data range:", data.index.min().strftime("%Y-%m-%d"), "→", data.index.max().strftime("%Y-%m-%d"))

# ---------- Normalize ----------
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[features])

# ---------- สร้าง Sequence ----------
forecast_horizon = 30
step = 1
seq_length_lstm = 300
seq_length_trans = 300

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


# ---------- สร้าง sequence สำหรับ LSTM ----------
X_lstm, y_lstm = create_sequences(scaled_features, close_prices, seq_length_lstm, forecast_horizon, step)
train_size_lstm = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:train_size_lstm], X_lstm[train_size_lstm:]
y_train_lstm, y_test_lstm = y_lstm[:train_size_lstm], y_lstm[train_size_lstm:]

# ---------- สร้าง sequence สำหรับ Transformer ----------
X_trans, y_trans = create_sequences(scaled_features, close_prices, seq_length_trans, forecast_horizon, step)
train_size_trans = int(len(X_trans) * 0.8)
X_train_trans, X_test_trans = X_trans[:train_size_trans], X_trans[train_size_trans:]
y_train_trans, y_test_trans = y_trans[:train_size_trans], y_trans[train_size_trans:]

# ---------- Callbacks ----------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    # tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)
]

# ---------- สร้างโมเดล LSTM ----------
def build_lstm_model():
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(seq_length_lstm, X_lstm.shape[2])),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(forecast_horizon)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------- สร้างโมเดล Transformer ----------
class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.pos_encoding = self.add_weight(name="pos_encoding", shape=(1, sequence_length, d_model),
                                            initializer="random_normal", trainable=True)

    def call(self, x):
        return x + self.pos_encoding

# Transformer Encoder Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(inputs.shape[-1])(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# โมเดล Transformer
def build_transformer_model():
    inputs = Input(shape=(seq_length_trans,  X_trans.shape[2]))
    x = LearnablePositionalEncoding(seq_length_trans,  X_trans.shape[2])(inputs)
    for _ in range(3):  # เพิ่ม depth
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(forecast_horizon)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model



# ---------- เทรนโมเดล ----------
lstm_model = build_lstm_model()
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=16, validation_split=0.1, callbacks=callbacks, verbose=0)

transformer_model = build_transformer_model()
transformer_model.fit(X_train_trans, y_train_trans, epochs=50, batch_size=16, validation_split=0.1, callbacks=callbacks, verbose=0)
# ---------- ประเมินผล ----------
def evaluate_model(model_name, y_pred, y_true, test_start_idx, seq_length):
    true_prices = []
    pred_prices = []
    for i in range(len(y_true)):
        base_price = close_prices[test_start_idx + seq_length + i - 1]
        true_seq = base_price * (1 + y_true[i])
        pred_seq = base_price * (1 + y_pred[i])
        true_prices.extend(true_seq)
        pred_prices.extend(pred_seq)
    mae = mean_absolute_error(true_prices, pred_prices)
    r2 = r2_score(true_prices, pred_prices)
    mape = np.mean(np.abs((np.array(true_prices) - np.array(pred_prices)) / np.array(true_prices))) * 100

    print(f"{model_name} MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
    return mae, r2, mape

evaluate_model("LSTM", lstm_model.predict(X_test_lstm), y_lstm[train_size_lstm:], train_size_lstm, seq_length_lstm)
evaluate_model("Transformer", transformer_model.predict(X_test_trans), y_trans[train_size_trans:], train_size_trans, seq_length_trans)

last_seq_lstm = X_lstm[-1]
last_seq_trans = X_trans[-1]
last_close = close_prices[-1]
last_date = data.index[-1]
future_dates = [last_date + timedelta(days=(i + 1) * step) for i in range(forecast_horizon)]

lstm_pred_pct = lstm_model.predict(last_seq_lstm.reshape(1, seq_length_lstm, X_lstm.shape[2]))[0]
trans_pred_pct = transformer_model.predict(last_seq_trans.reshape(1, seq_length_trans, X_trans.shape[2]))[0]

lstm_future_prices = last_close * (1 + lstm_pred_pct)
trans_future_prices = last_close * (1 + trans_pred_pct)


# --- คำนวณการเปลี่ยนแปลงเป็นเปอร์เซ็นต์ (บวกหรือลบ) ---
lstm_return_pct = ((lstm_future_prices[-1] - last_close) / last_close) * 100
trans_return_pct = ((trans_future_prices[-1] - last_close) / last_close) * 100

print(f"\n LSTM คาดการณ์ว่าอีก {forecast_horizon * step} วันจะ {'เพิ่มขึ้น📈' if lstm_return_pct >= 0 else 'ลดลง📉'} {lstm_return_pct:.2f}%")
print(f" Transformer คาดการณ์ว่าอีก {forecast_horizon * step} วันจะ {'เพิ่มขึ้น📈' if trans_return_pct >= 0 else 'ลดลง📉'} {trans_return_pct:.2f}%")

# --- เตรียมข้อมูลราคาจริงช่วง 60 วันย้อนหลังสำหรับกราฟ ---
lookback_days = 60
actual_dates = data.index[-lookback_days:]
actual_prices = close_prices[-lookback_days:]
lstm_model.save(f"lstm_{ticker}.h5")
transformer_model.save(f"transformer_{ticker}.h5")

# --- แสดงกราฟ ---
plt.figure(figsize=(14, 6))
plt.plot(actual_dates, actual_prices, label="Actual Prices", color='green')
plt.plot(future_dates, lstm_future_prices, label="LSTM Prediction", marker='o', color='orange')
plt.plot(future_dates, trans_future_prices, label="Transformer Prediction", marker='o', color='blue')
plt.axvline(x=last_date, color='gray', linestyle='--', label='Prediction Start')
plt.title(f"{ticker} Forecast Comparison: LSTM vs Transformer")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# ================= Loop หลาย seq_length สำหรับ LSTM และ Transformer =================
# seq_list = [90, 120, 150, 180, 240, 300]
#
# # Callbacks
# early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
# reduce_lr  = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=0)
#
# def create_sequences(features, close_prices, seq_len, horizon, step):
#     X, y = [], []
#     for i in range(len(features) - seq_len - horizon * step + 1):
#         X.append(features[i:i+seq_len])
#         base_price = close_prices[i + seq_len - 1]
#         targets = []
#         for j in range(1, horizon + 1):
#             future_price = close_prices[i + seq_len + j*step - 1]
#             targets.append((future_price - base_price) / base_price)
#         y.append(targets)
#     return np.array(X), np.array(y)
#
# # --- Encoder ที่ปลอดภัยกับ shape ---
# def transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.1, last_dim=None):
#     if last_dim is None:
#         last_dim = int(x.shape[-1])
#     attn = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
#     attn = Dropout(dropout)(attn)
#     x = Add()([x, attn])
#     x = LayerNormalization(epsilon=1e-6)(x)
#
#     ff = Dense(ff_dim, activation="relu")(x)
#     ff = Dense(last_dim)(ff)
#     x = Add()([x, ff])
#     x = LayerNormalization(epsilon=1e-6)(x)
#     return x
#
# class LearnablePositionalEncoding(tf.keras.layers.Layer):
#     def __init__(self, sequence_length, d_model):
#         super().__init__()
#         self.pos = self.add_weight(
#             name="pos_encoding",
#             shape=(1, sequence_length, d_model),
#             initializer="random_normal",
#             trainable=True,
#         )
#     def call(self, x):
#         return x + self.pos
#
# def build_lstm_model(seq_len, feat_dim, horizon):
#     m = Sequential([
#         Bidirectional(LSTM(64, return_sequences=True), input_shape=(seq_len, feat_dim)),
#         Dropout(0.2),
#         Bidirectional(LSTM(64)),
#         Dropout(0.2),
#         Dense(32, activation='relu'),
#         Dense(horizon)
#     ])
#     m.compile(optimizer='adam', loss='mse')
#     return m
#
# def build_transformer_model(seq_len, feat_dim, horizon):
#     inputs = Input(shape=(seq_len, feat_dim))
#     x = LearnablePositionalEncoding(seq_len, feat_dim)(inputs)
#     x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1, last_dim=feat_dim)
#     x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1, last_dim=feat_dim)
#     x = tf.keras.layers.GlobalAveragePooling1D()(x)
#     x = Dropout(0.2)(x)
#     x = Dense(64, activation='relu')(x)
#     out = Dense(horizon)(x)
#     m = Model(inputs, out)
#     m.compile(optimizer='adam', loss='mse')
#     return m
#
# def evaluate_model(y_pred, y_true, test_start_idx, seq_len, close_prices):
#     true_prices, pred_prices = [], []
#     for i in range(len(y_true)):
#         base = close_prices[test_start_idx + seq_len + i - 1]
#         true_seq = base * (1 + y_true[i])
#         pred_seq = base * (1 + y_pred[i])
#         true_prices.extend(true_seq)
#         pred_prices.extend(pred_seq)
#     mae  = mean_absolute_error(true_prices, pred_prices)
#     r2   = r2_score(true_prices, pred_prices)
#     mape = np.mean(np.abs((np.array(true_prices) - np.array(pred_prices)) / np.array(true_prices))) * 100
#     return mae, mape, r2
#
# results = []  # เก็บสรุปผลทุก seq_length / model
#
# for seq_len in seq_list:
#     # สร้างชุดข้อมูลตาม seq_len นี้
#     X, y = create_sequences(scaled_features, close_prices, seq_len, forecast_horizon, step)
#     if len(X) == 0:
#         print(f"⚠️ ข้าม seq_len={seq_len} (ข้อมูลไม่พอ)")
#         continue
#     feat_dim = X.shape[2]
#     train_size = int(len(X) * 0.8)
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]
#
#     # ---------- LSTM ----------
#     lstm = build_lstm_model(seq_len, feat_dim, forecast_horizon)
#     lstm.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1,
#              callbacks=[early_stop, reduce_lr], verbose=0)
#     y_pred_lstm = lstm.predict(X_test, verbose=0)
#     mae, mape, r2 = evaluate_model(y_pred_lstm, y_test, train_size, seq_len, close_prices)
#     results.append({"model":"LSTM", "seq_length": seq_len, "MAE": mae, "MAPE": mape, "R2": r2})
#
#     # ---------- Transformer ----------
#     trans = build_transformer_model(seq_len, feat_dim, forecast_horizon)
#     trans.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1,
#               callbacks=[early_stop, reduce_lr], verbose=0)
#     y_pred_trans = trans.predict(X_test, verbose=0)
#     mae, mape, r2 = evaluate_model(y_pred_trans, y_test, train_size, seq_len, close_prices)
#     results.append({"model":"Transformer", "seq_length": seq_len, "MAE": mae, "MAPE": mape, "R2": r2})
#
# # สรุปผลเป็นตาราง
# results_df = pd.DataFrame(results).sort_values(["model","seq_length"])
# print("\n==== Summary (MAE / MAPE / R²) ====")
# print(results_df.to_string(index=False))
#
# # เซฟเป็น CSV
# results_df.to_csv(f"{ticker}_seq_sweep_results.csv", index=False)
