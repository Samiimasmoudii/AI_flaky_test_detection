import pandas as pd

def load_and_clean_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Project URL', 'SHA Detected'])
    df = df[~df['Notes'].str.contains("deleted", case=False, na=False)]
    return df
