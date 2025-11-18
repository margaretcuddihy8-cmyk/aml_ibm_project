import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)

def load_transaction_data(file_path, verbose=True):

    """
    Loads the compressed transaction dataset.

    Parameters:
    - path (str): Path to the .csv.gz file.
    - nrows (int or None): Number of rows to read (for sampling or testing).

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")

    print(f"[INFO] Loading data from: {path}")
    df = pd.read_csv(path, compression="gzip", nrows=nrows)


    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.dropna(how='all', inplace=True)

    expected_cols = [
        "timestamp",
        "from_bank",
        "account",
        "to_bank",
        "account.1",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "is_laundering"
    ]

    missing = [col for col in expected_cols if col not in df.columns]

    if verbose:
        logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
        if missing:
            logging.warning(f"Missing expected columns: {missing}")
        logging.info(f"Columns loaded: {df.columns.tolist()}")

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Convert bank codes to categorical
    df['from_bank'] = df['from_bank'].astype('category')
    df['to_bank'] = df['to_bank'].astype('category')

    # Convert text features to categorical
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")


    return df