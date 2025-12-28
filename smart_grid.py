# =====================================================
# CitySense360 - Smart Grid Energy Forecasting (LSTM)
# Single-file end-to-end implementation
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
CSV_PATH = "smart_grid/smart_grid_dataset.csv"   # <-- your dataset
TIME_STEPS = 30               # lookback window
EPOCHS = 5
BATCH_SIZE = 32

# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------
print("Loading smart grid dataset...")

df = pd.read_csv(CSV_PATH)

# Parse timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index('Timestamp')

# Drop rows with missing values
df = df.dropna()

print("Dataset shape:", df.shape)

# -----------------------------------------------------
# 2. FEATURE SELECTION
# -----------------------------------------------------
# Target: Power Usage (kW)
target = 'Power Usage (kW)'

# Drop target from features
X = df.drop(columns=[target])
y = df[target]

# -----------------------------------------------------
# 3. NORMALIZATION
# -----------------------------------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# -----------------------------------------------------
# 4. CREATE TIME SERIES SEQUENCES
# -----------------------------------------------------
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

# -----------------------------------------------------
# 5. TRAIN / TEST SPLIT (TIME-AWARE)
# -----------------------------------------------------
split = int(0.8 * len(X_seq))

X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print("Train samples:", X_train.shape[0])
print("Test samples :", X_test.shape[0])

# -----------------------------------------------------
# 6. BUILD LSTM MODEL
# -----------------------------------------------------
model = Sequential([
    LSTM(128, return_sequences=True,
         input_shape=(TIME_STEPS, X_train.shape[2])),
    Dropout(0.3),

    LSTM(64),
    Dropout(0.3),

    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()

# -----------------------------------------------------
# 7. TRAIN MODEL
# -----------------------------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------------------------------
# 8. EVALUATION
# -----------------------------------------------------
y_pred = model.predict(X_test)

# Inverse scaling
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)

mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

print("\nEvaluation Metrics")
print("------------------")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# -----------------------------------------------------
# 9. VISUALIZATION
# -----------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv[:200], label="Actual Power Usage (kW)")
plt.plot(y_pred_inv[:200], label="Predicted Power Usage (kW)")
plt.title("Smart Grid Power Usage Forecasting")
plt.xlabel("Time Steps")
plt.ylabel("Power Usage (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 10. SAVE MODEL
# -----------------------------------------------------
model.save("smart_grid_lstm_model.h5")
print("\nModel saved as smart_grid_lstm_model.h5")
