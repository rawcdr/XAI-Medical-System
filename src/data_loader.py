import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print("Dataset Loaded. Shape:", df.shape)
    return df
    