# === Dependencies ===
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.optimizers.legacy import Adam  # faster on M1/M2

# === Config ===
LOOKBACK = 1
EPOCHS   = 20
BATCH    = 70
LR       = 5e-4
DROPOUT  = 0.2
USE_ALL_BINS = True

# === Helper functions ===
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def smape(y_true, y_pred, eps=1e-6):
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

def make_sequences(x, y, lookback):
    """Build sequences of length lookback to predict the next day (t+1)."""
    X_seq, y_seq = [], []
    for i in range(lookback, len(x) - 1):
        X_seq.append(x[i - lookback:i])
        y_seq.append(y[i + 1])  # predict next day
    return np.asarray(X_seq, dtype="float32"), np.asarray(y_seq, dtype="float32")

# === Load preprocessed data ===
csv_path = Path("data/processed_filllevel.csv")
df = pd.read_csv(csv_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["serialNumber", "date"]).reset_index(drop=True)

# === Feature engineering ===
def add_features(g):
    g = g.copy()
    g["dow"] = g["date"].dt.weekday
    g["month"] = g["date"].dt.month
    g["dow_sin"] = np.sin(2 * np.pi * g["dow"] / 7)
    g["dow_cos"] = np.cos(2 * np.pi * g["dow"] / 7)
    g["mon_sin"] = np.sin(2 * np.pi * g["month"] / 12)
    g["mon_cos"] = np.cos(2 * np.pi * g["month"] / 12)
    for L in [1, 2, 3, 7, 14]:
        g[f"lag_{L}"] = g["latestFullness"].shift(L)
    g["roll7_mean"] = g["latestFullness"].rolling(7).mean()
    g["roll7_std"]  = g["latestFullness"].rolling(7).std()
    g = g.dropna().reset_index(drop=True)
    return g

df_aug = df.groupby("serialNumber", group_keys=False).apply(add_features).reset_index(drop=True)
df_aug = df_aug.dropna().reset_index(drop=True)

# === Feature and target selection ===
feature_cols = [c for c in df_aug.columns if c not in 
                ["serialNumber", "date", "latitude", "longitude", "latestFullness"]]
target_col = "latestFullness"

print(f"Feature count: {len(feature_cols)} | Features: {feature_cols[:10]}...")

# === Select dataset ===
if USE_ALL_BINS:
    g = df_aug.copy()
else:
    bin_id = df_aug["serialNumber"].value_counts().idxmax()
    g = df_aug[df_aug["serialNumber"] == bin_id].reset_index(drop=True)
    print(f"Using single bin {bin_id} ({len(g)} records)")

# === Chronological split (80/20) ===
n = len(g)
split = int(0.8 * n)
X_raw, y_raw = g[feature_cols].values, g[target_col].values
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

# === Scaling (fit only on train) ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_s = scaler_X.fit_transform(X_train_raw)
X_test_s  = scaler_X.transform(X_test_raw)

y_train_s = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
y_test_s  = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()

# === Build sequences ===
Xtr, ytr = make_sequences(X_train_s, y_train_s, LOOKBACK)
Xte, yte = make_sequences(np.concatenate([X_train_s[-LOOKBACK:], X_test_s], axis=0),
                          np.concatenate([y_train_s[-LOOKBACK:], y_test_s], axis=0),
                          LOOKBACK)
print(f"Train {Xtr.shape}, Test {Xte.shape}")

# === Model builders ===
def build_lstm(input_dim):
    model = Sequential([
        LSTM(100, input_shape=(LOOKBACK, input_dim)),
        Dropout(DROPOUT),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LR), loss="mse",
              metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

    return model

def build_bilstm(input_dim):
    model = Sequential([
        Bidirectional(LSTM(100), input_shape=(LOOKBACK, input_dim)),
        Dropout(DROPOUT),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LR), loss="mae",
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name="RMSE")])
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

    # Predictions (test)
    yhat_s = model.predict(Xte).ravel()
    yhat = scaler_y.inverse_transform(yhat_s.reshape(-1, 1)).ravel()
    ytrue = scaler_y.inverse_transform(yte.reshape(-1, 1)).ravel()

    metrics = {
        "RMSE": rmse(ytrue, yhat),
        "MAE": mean_absolute_error(ytrue, yhat),
        "sMAPE": smape(ytrue, yhat),
        "R2": r2_score(ytrue, yhat)
    }

    print(f"\n{name} metrics: {metrics}")
    return metrics

# === Run both models ===
results = {}
results["LSTM"] = train_and_evaluate(build_lstm, "LSTM")
results["BiLSTM"] = train_and_evaluate(build_bilstm, "BiLSTM")

# === Summary ===
print("\n=== Final Comparison ===")
for k, v in results.items():
    print(f"{k:8s} | RMSE={v['RMSE']:.3f} | MAE={v['MAE']:.3f} | sMAPE={v['sMAPE']:.2f}% | R2={v['R2']:.3f}")