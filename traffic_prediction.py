# =====================================================
# CitySense360 - Traffic Congestion Prediction
# CCTV Video → Vehicle Detection → LSTM Forecast
# Single-file end-to-end implementation
# =====================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
VIDEO_DIR = "traffic/video"
FRAME_SKIP = 10          # process every Nth frame
TIME_STEPS = 12          # lookback window
EPOCHS = 20
BATCH_SIZE = 32

# COCO vehicle classes
VEHICLE_CLASSES = [2, 3, 5, 7]  
# 2=car, 3=motorbike, 5=bus, 7=truck

# -----------------------------------------------------
# 1. LOAD YOLO MODEL (PRETRAINED)
# -----------------------------------------------------
print("Loading YOLO model...")
yolo = YOLO("yolov8n.pt")  # lightweight & fast

# -----------------------------------------------------
# 2. PROCESS VIDEOS → TRAFFIC COUNTS
# -----------------------------------------------------
print("Processing CCTV videos...")

traffic_data = []

for video_file in os.listdir(VIDEO_DIR):
    if not video_file.endswith(".avi"):
        continue

    video_path = os.path.join(VIDEO_DIR, video_file)
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    timestamp = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_SKIP == 0:
            results = yolo(frame, verbose=False)[0]

            vehicle_count = 0
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls in VEHICLE_CLASSES:
                    vehicle_count += 1

            traffic_data.append({
                "time_index": timestamp,
                "vehicle_count": vehicle_count
            })

            timestamp += 1

        frame_id += 1

    cap.release()

traffic_df = pd.DataFrame(traffic_data)
print("Traffic samples collected:", len(traffic_df))

# -----------------------------------------------------
# 3. PREPARE TIME SERIES DATA
# -----------------------------------------------------
X = traffic_df[['vehicle_count']].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

def create_sequences(data, time_steps):
    Xs, ys = [], []
    for i in range(len(data) - time_steps):
        Xs.append(data[i:i + time_steps])
        ys.append(data[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, TIME_STEPS)

# -----------------------------------------------------
# 4. TRAIN / TEST SPLIT
# -----------------------------------------------------
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print("Train samples:", X_train.shape[0])
print("Test samples :", X_test.shape[0])

# -----------------------------------------------------
# 5. BUILD LSTM MODEL
# -----------------------------------------------------
model = Sequential([
    LSTM(64, return_sequences=True,
         input_shape=(TIME_STEPS, 1)),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()

# -----------------------------------------------------
# 6. TRAIN MODEL
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
# 7. EVALUATION
# -----------------------------------------------------
y_pred = model.predict(X_test)

y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

print("\nEvaluation Metrics")
print("------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# -----------------------------------------------------
# 8. VISUALIZATION
# -----------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv[:200], label="Actual Traffic Density")
plt.plot(y_pred_inv[:200], label="Predicted Traffic Density")
plt.title("Traffic Congestion Prediction (CCTV + LSTM)")
plt.xlabel("Time Index")
plt.ylabel("Vehicle Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 9. SAVE MODEL
# -----------------------------------------------------
model.save("traffic_lstm_model.h5")
print("\nModel saved as traffic_lstm_model.h5")
