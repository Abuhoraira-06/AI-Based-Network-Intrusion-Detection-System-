import pandas as pd
import numpy as np

def load_data(use_real_data=False, file_path=None):
    if use_real_data:
        if file_path is None:
            raise ValueError("File path required for real data")

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        required_columns = [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Packet Length Mean",
            "Active Mean",
            "Label"
        ]

        df = df[required_columns]
        df["Label"] = df["Label"].apply(
            lambda x: 0 if str(x).upper() == "BENIGN" else 1
        )

        return df

    np.random.seed(42)
    samples = 5000

    return pd.DataFrame({
        "Destination Port": np.random.randint(1, 65535, samples),
        "Flow Duration": np.random.randint(100, 100000, samples),
        "Total Fwd Packets": np.random.randint(1, 200, samples),
        "Packet Length Mean": np.random.uniform(10, 1500, samples),
        "Active Mean": np.random.uniform(0, 1000, samples),
        "Label": np.random.choice([0, 1], samples, p=[0.7, 0.3])
    })