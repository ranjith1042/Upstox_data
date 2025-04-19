import pandas as pd
import os
from datetime import datetime

def get_latest_instruments_file():
    """Get the most recent instruments file from the data directory"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Data directory not found. Please run download_instruments.py first.")
    
    files = [f for f in os.listdir(data_dir) if f.startswith("nse_instruments_") and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No instruments files found. Please run download_instruments.py first.")
    
    # Get the most recent file
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
    return os.path.join(data_dir, latest_file)

def read_symbols_from_file():
    """Read symbols from symbols.tls file"""
    try:
        with open("symbols.tls", "r") as f:
            symbols = [line.strip() for line in f.readlines() if line.strip()]
        return symbols
    except FileNotFoundError:
        raise FileNotFoundError("symbols.tls file not found. Please create it with one symbol per line.")

def get_instrument_keys(symbols, segment="NSE_EQ", instrument_type="EQ"):
    """
    Get instrument keys for the given symbols
    
    Args:
        symbols (list): List of trading symbols
        segment (str): Market segment (default: "NSE_EQ")
        instrument_type (str): Instrument type (default: "EQ")
    
    Returns:
        dict: Dictionary mapping symbols to their instrument keys
    """
    # Get the latest instruments file
    instruments_file = get_latest_instruments_file()
    print(f"Using instruments file: {instruments_file}")
    
    # Read the instruments data
    df = pd.read_csv(instruments_file)
    
    # Filter for the specified segment and instrument type
    filtered_df = df[
        (df['segment'] == segment) & 
        (df['instrument_type'] == instrument_type)
    ]
    
    # Create a dictionary to store results
    instrument_keys = {}
    not_found = []
    
    # Find instrument keys for each symbol
    for symbol in symbols:
        # Clean the symbol (remove .NS if present)
        clean_symbol = symbol.replace('.NS', '')
        
        # Find matching instrument
        match = filtered_df[filtered_df['trading_symbol'] == clean_symbol]
        
        if not match.empty:
            instrument_keys[symbol] = match.iloc[0]['instrument_key']
        else:
            not_found.append(symbol)
    
    # Print results
    print("\nFound instrument keys:")
    for symbol, key in instrument_keys.items():
        print(f"{symbol}: {key}")
    
    if not_found:
        print("\nSymbols not found:")
        for symbol in not_found:
            print(symbol)
    
    return instrument_keys

def save_instrument_keys(instrument_keys):
    """Save instrument keys to a CSV file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/instrument_keys_{timestamp}.csv"
    
    # Convert to DataFrame and save
    df = pd.DataFrame(list(instrument_keys.items()), columns=['symbol', 'instrument_key'])
    df.to_csv(output_file, index=False)
    print(f"\nSaved instrument keys to {output_file}")

def main():
    try:
        # Read symbols from file
        symbols = read_symbols_from_file()
        print(f"Found {len(symbols)} symbols in symbols.tls")
        
        # Get instrument keys
        instrument_keys = get_instrument_keys(symbols)
        
        # Save results
        save_instrument_keys(instrument_keys)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 