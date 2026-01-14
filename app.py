# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import math
import joblib


app = Flask(__name__, template_folder="templates")

# ================= CONFIG =================
MODEL_PATH = "random_forest_model.pkl"
ACC_SENS = 16384      # LSB/g
GYRO_SENS = 131       # LSB/(°/s)
G_TO_MS2 = 9.80665
# =========================================

# ===== Load model =====
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model .pkl tidak ditemukan")

model = joblib.load(MODEL_PATH)


# ======================================================
# Convert RAW 3 MPU6050 -> feature vector
# ======================================================
def prepare_features_3mpu(data):
    """
    data:
    ax1 ay1 az1 gx1 gy1 gz1
    ax2 ay2 az2 gx2 gy2 gz2
    ax3 ay3 az3 gx3 gy3 gz3
    """

    sensors = ["1", "2", "3"]
    features = []

    for s in sensors:
        ax = float(data[f"ax{s}"])
        ay = float(data[f"ay{s}"])
        az = float(data[f"az{s}"])
        gx = float(data[f"gx{s}"])
        gy = float(data[f"gy{s}"])
        gz = float(data[f"gz{s}"])

        # Accelerometer → m/s²
        features.extend([
            (ax / ACC_SENS) * G_TO_MS2,
            (ay / ACC_SENS) * G_TO_MS2,
            (az / ACC_SENS) * G_TO_MS2
        ])

        # Gyroscope → rad/s
        features.extend([
            (gx / GYRO_SENS) * (math.pi / 180),
            (gy / GYRO_SENS) * (math.pi / 180),
            (gz / GYRO_SENS) * (math.pi / 180)
        ])

    X = np.array([features], dtype=float)


# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # nanti ini diganti hasil model
        result = {
            "meja": 2,
            "mulut": 3,
            "kepala_depan": 1,
            "kepala_belakang": 2
        }

        total = sum(result.values())

        return jsonify({
            "scores": result,
            "total": total
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400



# ================= RUN ====================
if __name__ == "__main__":
    app.run(debug=True)
