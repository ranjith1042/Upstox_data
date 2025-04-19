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

def get_stock_data(symbol, to_date):
    """Fetch stock data from Upstox API in reverse chronological order"""
    try:
        # Get instrument keys
        instrument_keys_df = get_latest_instrument_keys()
        
        # Find the instrument key for this symbol
        symbol_data = instrument_keys_df[instrument_keys_df['symbol'] == symbol]
        if symbol_data.empty:
            print(f"Instrument key not found for symbol: {symbol}")
            return None
            
        instrument_key = symbol_data['instrument_key'].iloc[0]
        
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {ACCESS_TOKEN}'
        }
        
        # Format dates for API
        if isinstance(to_date, datetime):
            to_date = to_date.date()
        
        all_data = []
        current_to_date = to_date
        
        while True:
            try:
                # Calculate from_date (10 years before current_to_date)
                from_date = current_to_date - timedelta(days=3650)
                
                # Format dates as strings
                from_date_str = from_date.strftime('%Y-%m-%d')
                to_date_str = current_to_date.strftime('%Y-%m-%d')
                
                # API endpoint for historical data - using day interval
                url = f"{BASE_URL}/historical-candle/{instrument_key}/days/1/{to_date_str}/{from_date_str}"
                
                print(f"Requesting data from {from_date_str} to {to_date_str}")
                
                response = requests.get(url, headers=headers)
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success' and data.get('data', {}).get('candles'):
                        candles = data['data']['candles']
                        if candles:
                            all_data.extend(candles)
                            print(f"Retrieved {len(candles)} records")
                        else:
                            print("No data in this period")
                            break
                    else:
                        print("API returned error status")
                        break
                else:
                    print(f"API request failed with status {response.status_code}")
                    break
                
                # Move to_date back by 10 years for next iteration
                current_to_date = from_date
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                break
        
        if all_data:
            df = pd.DataFrame(all_data, 
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            # Convert timestamp to IST
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(IST)
            # Sort by timestamp in ascending order
            df = df.sort_values('timestamp')
            print(f"Total records retrieved: {len(df)}")
            return df
        else:
            print(f"No data found for {symbol}")
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
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Stored {len(df)} records for {symbol}")

def get_symbols_from_file():
    """Read symbols from symbols.tls file"""
    try:
        with open('symbols.tls', 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        return symbols
    except FileNotFoundError:
        print("symbols.tls file not found")
        return []

def main():
    try:
        # Get symbols from file
        symbols = get_symbols_from_file()
        if not symbols:
            print("No symbols found in symbols.tls")
            return
            
        print(f"Processing {len(symbols)} symbols")
        
        # Set end date
        end_date = datetime.strptime('2025-04-17', '%Y-%m-%d').date()
        print(f"Fetching data up to: {end_date.strftime('%Y-%m-%d')}")
        
        # Fetch and store data for each symbol
        for i, symbol in enumerate(symbols, 1):
            print(f"\nProcessing {i}/{len(symbols)}: {symbol}...")
            df = get_stock_data(symbol, end_date)
            if df is not None:
                store_data_in_db(symbol, df)
        
        print("\nData download and storage completed!")
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main() 