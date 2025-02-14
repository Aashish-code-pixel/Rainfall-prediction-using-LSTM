import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("data/rainfall_data.csv", parse_dates=["Date"], index_col="Date")

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[["Rainfall"]])

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 30  # Use last 30 days to predict next
X, y = create_sequences(df_scaled, SEQ_LENGTH)

# Save processed data
np.save("data/X.npy", X)
np.save("data/y.npy", y)
