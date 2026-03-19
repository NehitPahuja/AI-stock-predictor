import pandas as pd
import numpy as np

def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean historical stock data by handling missing values.
    Specifically:
    - Drops rows where all elements are NaN
    - Performs forward fill for missing values (e.g., trading halts)
    - Performs backward fill for any remaining NaNs at the beginning
    
    Args:
        df (pd.DataFrame): The raw dataframe fetched from yfinance.
        
    Returns:
        pd.DataFrame: A cleaned dataframe without NaN values in core columns.
    """
    if df.empty:
        print("Warning: Empty DataFrame provided for cleaning.")
        return df
        
    # Make a copy so we don't unexpectedly mutate the original dataframe
    df_clean = df.copy()
    
    # 1. Drop rows where ALL elements are missing (completely empty trading days)
    df_clean.dropna(how='all', inplace=True)
    
    # 2. Forward fill: use previous day's data if current is missing
    df_clean.ffill(inplace=True)
    
    # 3. Backward fill: use next day's data for any missing values at the beginning of the series
    df_clean.bfill(inplace=True)
    
    # Optional: Fill any remaining obscure NaNs (e.g., if entire column is NaN) with 0
    df_clean.fillna(0, inplace=True)
    
    # Ensure correct fundamental data types
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
    print(f"Data cleaning complete. Shape: {df_clean.shape}")
    return df_clean

if __name__ == "__main__":
    # Test our data cleaning with a mock dataframe containing missing values
    print("--- Testing Data Cleaning Module ---")
    mock_data = pd.DataFrame({
        'Open': [100, np.nan, 102, 105],
        'High': [105, 106, np.nan, 110],
        'Low': [95, 96, 98, np.nan],
        'Close': [102, 104, 103, 108],
        'Volume': [1000, np.nan, 1200, 1500]
    })
    
    print("Original Data (with NaNs):")
    print(mock_data)
    
    cleaned_data = clean_stock_data(mock_data)
    print("\nCleaned Data (after ffill/bfill):")
    print(cleaned_data)
