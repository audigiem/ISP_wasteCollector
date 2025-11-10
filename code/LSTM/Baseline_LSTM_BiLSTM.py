# === Dependencies ===
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers.legacy import Adam  # faster on M1/M2

# === Config ===
EPOCHS = 20
BATCH = 70
LR = 5e-4
DROPOUT = 0.2
LOOKBACK = 1  # baseline = predict next day from current day

# === Helper metrics ===
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps=1e-6):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

# === Load dataset ===
csv_path = Path("data/processed_filllevel.csv")
df = pd.read_csv(csv_path)

# Ensure correct sorting
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["serialNumber", "date"]).reset_index(drop=True)

# === Select only required columns ===
df_base = df[["serialNumber", "date", "latestFullness"]].dropna().reset_index(drop=True)

# Use all bins combined (as done in paper)
g = df_base.copy()
print(f"Using {g['serialNumber'].nunique()} bins, total {len(g)} records.")

# === Chronological split (80/20) ===
n = len(g)
split = int(0.8 * n)

X_raw = g[["latestFullness"]].values
y_raw = g["latestFullness"].values

X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

# === Scaling ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_s = scaler_X.fit_transform(X_train_raw)
X_test_s  = scaler_X.transform(X_test_raw)

y_train_s = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
y_test_s  = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()

# === Prepare sequences (lookback = 1 for baseline) ===
def make_sequences(x, y, lookback):
    X_seq, y_seq = [], []
    for i in range(lookback, len(x)):
        X_seq.append(x[i - lookback:i])
        y_seq.append(y[i])
    return np.asarray(X_seq, dtype="float32"), np.asarray(y_seq, dtype="float32")

Xtr, ytr = make_sequences(X_train_s, y_train_s, LOOKBACK)
Xte, yte = make_sequences(np.concatenate([X_train_s[-LOOKBACK:], X_test_s], axis=0),
                          np.concatenate([y_train_s[-LOOKBACK:], y_test_s], axis=0),
                          LOOKBACK)

print(f"Train shape: {Xtr.shape}, Test shape: {Xte.shape}")

# === Model builders ===
def build_lstm(input_dim):
    model = Sequential([
        LSTM(100, input_shape=(LOOKBACK, input_dim)),
        Dropout(DROPOUT),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LR),
                  loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE"),
                           tf.keras.metrics.RootMeanSquaredError(name="RMSE")])
    return model

def build_bilstm(input_dim):
    model = Sequential([
        Bidirectional(LSTM(100), input_shape=(LOOKBACK, input_dim)),
        Dropout(DROPOUT),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LR),
                  loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE"),
                           tf.keras.metrics.RootMeanSquaredError(name="RMSE")])
    return model

# === Training and evaluation ===
def train_and_evaluate(model_fn, name):
    print(f"\n=== Training {name} ===")
    model = model_fn(Xtr.shape[-1])
    model.summary()

    history = model.fit(
        Xtr, ytr,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1,
        shuffle=False
    )

    # Predict on test data
    yhat_s = model.predict(Xte).ravel()
    yhat = scaler_y.inverse_transform(yhat_s.reshape(-1, 1)).ravel()
    ytrue = scaler_y.inverse_transform(yte.reshape(-1, 1)).ravel()

    metrics = {
        "RMSE": rmse(ytrue, yhat),
        "MAE": mean_absolute_error(ytrue, yhat),
        "MAPE": mape(ytrue, yhat),
        "R2": r2_score(ytrue, yhat)
    }

    print(f"\n{name} metrics: {metrics}")
    return metrics

# === Run both models ===
results = {}
results["LSTM"] = train_and_evaluate(build_lstm, "LSTM")
results["BiLSTM"] = train_and_evaluate(build_bilstm, "BiLSTM")

# === Final summary ===
print("\n=== Final Comparison ===")
for k, v in results.items():
    print(f"{k:8s} | RMSE={v['RMSE']:.3f} | MAE={v['MAE']:.3f} | MAPE={v['MAPE']:.2f}% | R2={v['R2']:.3f}")
