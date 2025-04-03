import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import multi_country_pipeline as mcp
from tensorflow.keras.models import load_model
import tempfile
import os

# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="COVID-19 Forecast", layout="centered")
st.title("🦠 COVID-19 Forecast by Country")

# -------------------------------
# Sidebar input
# -------------------------------
countries = ['United States', 'New Zealand', 'India', 'Brazil']
selected_country = st.selectbox("Select a country to forecast:", countries)

# loading data
st.write("\n## Loading and preprocessing data...")
data = mcp.load_owid_data()
preprocessor = mcp.MultiCountryCOVIDPreprocessor(countries=countries, window_size=14) # pre processing data then fitting it
preprocessor.fit(data)
X, y = preprocessor.transform(data)

# building and training model
model = mcp.build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
model.fit(X, y, epochs=10, batch_size=32, verbose=0)

# predictions for the selected countru
st.write("\n## Predicting for: ", selected_country)

# get scaled data only for that country
scaled_data = mcp.prepare_country_data(data, selected_country, preprocessor.scaler)
encoded_data = mcp.encode_country(scaled_data.reset_index(), all_countries=countries) # all countries so dimensions match when only choosing one (one hot encoding)
X_pred, y_true = mcp.create_sliding_windows(encoded_data, window_size=14)
y_pred = model.predict(X_pred).flatten()

# -------------------------------
# Plotting
# -------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_true, label="Actual")
ax.plot(y_pred, label="Predicted")
ax.set_title(f"LSTM Forecast: {selected_country}")
ax.set_xlabel("Days")
ax.set_ylabel("Normalized New Cases")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# -------------------------------
# Download Model Button (Optional)
# -------------------------------
with tempfile.TemporaryDirectory() as tmp:
    model_path = os.path.join(tmp, "lstm_model.h5")
    model.save(model_path)
    with open(model_path, "rb") as f:
        st.download_button("Download Trained Model", f, file_name="lstm_model.h5")