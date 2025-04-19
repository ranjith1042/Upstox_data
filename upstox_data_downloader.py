import requests
import pandas as pd
from sqlalchemy import create_engine, Table, Column, String, Float, DateTime, MetaData
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
import urllib.parse
import pytz
import glob

# Load environment variables
load_dotenv()

# Upstox API configuration
API_KEY = os.getenv("UPSTOX_API_KEY")
API_SECRET = os.getenv("UPSTOX_API_SECRET")
ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
BASE_URL = "https://api.upstox.com/v3"

if not all([API_KEY, API_SECRET, ACCESS_TOKEN]):
    raise ValueError("Please set UPSTOX_API_KEY, UPSTOX_API_SECRET and UPSTOX_ACCESS_TOKEN environment variables")

# Database configuration
DB_PATH = "stock_data.db"
engine = create_engine(f'sqlite:///{DB_PATH}')
metadata = MetaData()

# Set IST timezone
IST = pytz.timezone('Asia/Kolkata')

def get_latest_instrument_keys():
    """Get the most recent instrument keys file"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Data directory not found. Please run get_instrument_keys.py first.")
    
    files = [f for f in os.listdir(data_dir) if f.startswith("instrument_keys_") and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No instrument keys files found. Please run get_instrument_keys.py first.")
    
    # Get the most recent file
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
    return pd.read_csv(os.path.join(data_dir, latest_file))

def create_stock_table(symbol):
    """Create a table for a specific stock if it doesn't exist"""
    # Remove .NS and special characters for table name
    clean_symbol = symbol.replace('.NS', '').replace('^', '').replace('-', '_')
    table_name = clean_symbol  # Removed 'stock_' prefix
    
    # Define the table structure
    table = Table(
        table_name,
        metadata,
        Column('timestamp', DateTime, primary_key=True),
        Column('open', Float),
        Column('high', Float),
        Column('low', Float),
        Column('close', Float),
        Column('volume', Float),
        Column('oi', Float)
    )
    
    # Create the table if it doesn't exist
    table.create(engine, checkfirst=True)
    return table

def get_stock_data(symbol, from_date, to_date):
    """Fetch stock data from Upstox API"""
    try:
        # Get instrument keys
        instrument_keys_df = get_latest_instrument_keys()
        
        # Find the instrument key for this symbol
        instrument_key = instrument_keys_df[instrument_keys_df['symbol'] == symbol]['instrument_key'].iloc[0]
        
        if not instrument_key:
            print(f"Instrument key not found for symbol: {symbol}")
            return None
            
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {ACCESS_TOKEN}'
        }
        
        # Format dates for API
        if isinstance(from_date, datetime):
            from_date = from_date.date()
        if isinstance(to_date, datetime):
            to_date = to_date.date()
        
        # Format dates as strings
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')
        
        # API endpoint for historical data - using day interval
        url = f"{BASE_URL}/historical-candle/{instrument_key}/days/1/{to_date_str}/{from_date_str}"
        
        print(f"Requesting URL: {url}")  # Debug print
        
        response = requests.get(url, headers=headers)
        print(f"Response status: {response.status_code}")  # Debug print
        
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'success':
            df = pd.DataFrame(data['data']['candles'], 
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            # Convert timestamp to IST (timestamps are already UTC-aware)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(IST)
            return df
        else:
            print(f"Error fetching data for {symbol}: {data.get('message', 'Unknown error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Error response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Error processing data for {symbol}: {str(e)}")
        return None

def store_data_in_db(symbol, df):
    """Store the data in the database"""
    if df is None or df.empty:
        return
    
    # Create table for the symbol
    table = create_stock_table(symbol)
    
    # Convert numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'oi']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])
    
    # Store data in database
    clean_symbol = symbol.replace('.NS', '').replace('^', '').replace('-', '_')
    table_name = clean_symbol  # Removed 'stock_' prefix
    df.to_sql(table_name, engine, if_exists='append', index=False)
    print(f"Stored {len(df)} records for {symbol}")

def main():
    try:
        # Get instrument keys
        instrument_keys_df = get_latest_instrument_keys()
        symbols = instrument_keys_df['symbol'].tolist()
        
        # Test with first 5 symbols
        test_symbols = symbols[:5]
        print(f"Testing with symbols: {test_symbols}")
        
        # Set fixed date range as specified
        start_date = datetime.strptime('2025-01-01', '%Y-%m-%d').date()
        end_date = datetime.strptime('2025-04-17', '%Y-%m-%d').date()
        
        print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Fetch and store data for each symbol
        for symbol in test_symbols:
            print(f"\nProcessing {symbol}...")
            df = get_stock_data(symbol, start_date, end_date)
            if df is not None:
                store_data_in_db(symbol, df)
        
        print("\nData download and storage completed!")
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main() 