import yfinance as yf
import pandas as pd
from typing import Optional

def fetch_stock_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical stock data using Yahoo Finance.
    Supports US stocks (e.g., 'AAPL') and Indian stocks (e.g., 'RELIANCE.NS', 'TCS.NS').
    
    Args:
        ticker (str): The stock symbol.
        start_date (str, optional): Start date in 'YYYY-MM-DD' format.
        end_date (str, optional): End date in 'YYYY-MM-DD' format.
        period (str): Time period (e.g., '1mo', '3mo', '1y', 'max'). Used if dates aren't provided.
        interval (str): Time interval (e.g., '1d', '1wk', '1mo').
        
    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """
    print(f"Fetching data for {ticker}...")
    
    try:
        stock = yf.Ticker(ticker)
        
        # If explicit dates are provided, use them. Otherwise, rely on the period.
        if start_date and end_date:
            df = stock.history(start=start_date, end=end_date, interval=interval)
        else:
            df = stock.history(period=period, interval=interval)
            
        if df.empty:
            print(f"Warning: No data found for ticker {ticker}.")
            return pd.DataFrame()
            
        # Optional: Save raw data to a local CSV for inspection
        filename = f"data/{ticker.replace('.', '_')}_raw.csv"
        df.to_csv(filename)
        print(f"Successfully fetched {len(df)} records for {ticker}. Saved to {filename}.")
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the loader with a US and Indian stock to verify Phase 1 requirements
    print("--- Testing Data Loader ---")
    
    aapl_data = fetch_stock_data("AAPL", period="1mo")
    if not aapl_data.empty:
        print(aapl_data.head(2))
        
    print("\n")
    
    tcs_data = fetch_stock_data("TCS.NS", period="1mo")
    if not tcs_data.empty:
        print(tcs_data.head(2))
