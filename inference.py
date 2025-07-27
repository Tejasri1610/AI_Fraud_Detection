import numpy as np
import joblib
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load models
xgb_model = joblib.load("xgboost_model.pkl")
isolation_model = joblib.load("isolation_forest.pkl")
meta_model = joblib.load("meta_model.pkl")
scaler = joblib.load("scaler.pkl")
cnn_model = tf.keras.models.load_model("cnn_model.h5")

def predict_fraud(features):
    features = np.array(features).reshape(1, -1)

    # Scale the input features
    features_scaled = scaler.transform(features)

    # CNN input shape must match what it was trained with
    cnn_input = features_scaled.reshape((features_scaled.shape[0], features_scaled.shape[1], 1))

    # Individual model predictions
    xgb_pred = xgb_model.predict(features_scaled)
    iso_pred = isolation_model.predict(features_scaled)
    iso_pred = np.where(iso_pred == 1, 0, 1)  # 1 means anomaly (fraud)

    cnn_pred = (cnn_model.predict(cnn_input, verbose=0) > 0.5).astype(int)

    # Stack predictions: combine all into single vector
    stacked_input = np.array([[xgb_pred[0], iso_pred[0], cnn_pred[0][0]]])

    # Final prediction from meta-model
    final_pred = meta_model.predict(stacked_input)

    return final_pred[0]
