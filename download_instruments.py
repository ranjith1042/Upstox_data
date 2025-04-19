import requests
import gzip
import json
import pandas as pd
from datetime import datetime
import os

def download_instruments():
    """
    Download NSE instruments data from Upstox and save it to a file.
    """
    url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    
    try:
        # Create a directory for storing the data if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Download the gzipped file
        print("Downloading instruments data...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Decompress the gzipped data
        decompressed_data = gzip.decompress(response.content)
        
        # Parse the JSON data
        instruments_data = json.loads(decompressed_data)
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(instruments_data)
        
        # Save the data to a CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"data/nse_instruments_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        
        print(f"Successfully downloaded and saved {len(df)} instruments to {csv_filename}")
        
        # Print some basic statistics
        print("\nBasic Statistics:")
        print(f"Total number of instruments: {len(df)}")
        print("\nExchange distribution:")
        print(df['exchange'].value_counts())
        print("\nInstrument types:")
        print(df['instrument_type'].value_counts())
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    download_instruments() 