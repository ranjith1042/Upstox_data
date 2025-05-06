import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from datetime import datetime
import sqlite3
from sqlalchemy import create_engine, text
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('screener_scraper.log'),
        logging.StreamHandler()
    ]
)

# Constants
BASE_URL = "https://www.screener.in/company/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_symbols_from_file():
    """Read symbols from symbols.tls file"""
    try:
        with open('symbols.tls', 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        return symbols
    except FileNotFoundError:
        logging.error("symbols.tls file not found")
        return []

def create_stock_tables(engine, symbol):
    """Create tables for a specific stock's financial statements"""
    clean_symbol = symbol.replace('.NS', '').replace('^', '').replace('-', '_')
    
    # Table definitions
    tables = {
        f'IS_{clean_symbol}_Q': '''
            CREATE TABLE IF NOT EXISTS IS_{clean_symbol}_Q (
                quarter TEXT PRIMARY KEY,
                revenue REAL,
                expenses REAL,
                operating_profit REAL,
                net_profit REAL,
                eps REAL
            )
        ''',
        f'IS_{clean_symbol}_A': '''
            CREATE TABLE IF NOT EXISTS IS_{clean_symbol}_A (
                year TEXT PRIMARY KEY,
                revenue REAL,
                expenses REAL,
                operating_profit REAL,
                net_profit REAL,
                eps REAL
            )
        ''',
        f'BS_{clean_symbol}_A': '''
            CREATE TABLE IF NOT EXISTS BS_{clean_symbol}_A (
                year TEXT PRIMARY KEY,
                total_assets REAL,
                total_liabilities REAL,
                total_equity REAL
            )
        ''',
        f'CF_{clean_symbol}_A': '''
            CREATE TABLE IF NOT EXISTS CF_{clean_symbol}_A (
                year TEXT PRIMARY KEY,
                operating_cash_flow REAL,
                investing_cash_flow REAL,
                financing_cash_flow REAL,
                net_cash_flow REAL
            )
        '''
    }
    
    with engine.connect() as conn:
        for table_name, create_query in tables.items():
            conn.execute(text(create_query.format(clean_symbol=clean_symbol)))
            conn.commit()
    
    return clean_symbol

def extract_financial_data(soup, statement_type):
    """Extract financial data from BeautifulSoup object based on statement type"""
    data = []
    try:
        # Find the relevant table based on statement type
        table = soup.find('table', {'id': f'{statement_type}-table'})
        if not table:
            return data
        
        # Extract headers and data rows
        headers = [th.text.strip() for th in table.find_all('th')]
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cells = row.find_all('td')
            if cells:
                row_data = [cell.text.strip() for cell in cells]
                data.append(dict(zip(headers, row_data)))
                
    except Exception as e:
        logging.error(f"Error extracting {statement_type} data: {str(e)}")
    
    return data

def convert_symbol_for_screener(symbol):
    """Convert symbol format for Screener.in URL"""
    # Remove .NS suffix and ^ prefix
    symbol = symbol.replace('.NS', '').replace('^', '')
    # Convert to uppercase
    symbol = symbol.upper()
    return symbol

def scrape_company_data(symbol):
    """Scrape financial data for a given symbol"""
    screener_symbol = convert_symbol_for_screener(symbol)
    url = f"{BASE_URL}{screener_symbol}/consolidated/"
    try:
        logging.info(f"Requesting URL: {url}")
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check if we got a valid response with data
        if "Page not found" in soup.text:
            logging.error(f"Page not found for symbol {symbol}")
            # Try without consolidated
            url = f"{BASE_URL}{screener_symbol}/"
            logging.info(f"Retrying with URL: {url}")
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if "Page not found" in soup.text:
                logging.error(f"Page not found for symbol {symbol} without consolidated")
                return None
        
        # Extract different types of financial data
        quarterly_pl = extract_financial_data(soup, 'quarters')  # Updated table ID
        annual_pl = extract_financial_data(soup, 'profit-loss')  # Updated table ID
        balance_sheet = extract_financial_data(soup, 'balance-sheet')
        cash_flow = extract_financial_data(soup, 'cash-flow')
        
        return {
            'quarterly_pl': quarterly_pl,
            'annual_pl': annual_pl,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow
        }
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping data for {symbol}: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error scraping data for {symbol}: {str(e)}")
        return None

def store_data_in_db(engine, symbol, data):
    """Store scraped data in the database"""
    if not data:
        return
    
    try:
        clean_symbol = create_stock_tables(engine, symbol)
        
        with engine.connect() as conn:
            # Store quarterly P/L data
            if data['quarterly_pl']:
                df = pd.DataFrame(data['quarterly_pl'])
                df.to_sql(f'IS_{clean_symbol}_Q', conn, if_exists='replace', index=False)
            
            # Store annual P/L data
            if data['annual_pl']:
                df = pd.DataFrame(data['annual_pl'])
                df.to_sql(f'IS_{clean_symbol}_A', conn, if_exists='replace', index=False)
            
            # Store balance sheet data
            if data['balance_sheet']:
                df = pd.DataFrame(data['balance_sheet'])
                df.to_sql(f'BS_{clean_symbol}_A', conn, if_exists='replace', index=False)
            
            # Store cash flow data
            if data['cash_flow']:
                df = pd.DataFrame(data['cash_flow'])
                df.to_sql(f'CF_{clean_symbol}_A', conn, if_exists='replace', index=False)
                
    except Exception as e:
        logging.error(f"Error storing data for {symbol}: {str(e)}")

def cleanup_old_tables(engine):
    """Remove old tables from previous runs"""
    with engine.connect() as conn:
        # Get all tables
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result]
        
        # Drop old tables
        old_tables = ['quarterly_pl', 'annual_pl', 'balance_sheet', 'cash_flow']
        for table in old_tables:
            if table in tables:
                conn.execute(text(f"DROP TABLE {table}"))
                conn.commit()
                logging.info(f"Dropped old table: {table}")

def main():
    # Create database
    engine = create_engine('sqlite:///fundamental_data.db')
    
    # Clean up old tables
    cleanup_old_tables(engine)
    
    # Get symbols
    symbols = get_symbols_from_file()
    if not symbols:
        return
    
    # Test with first 5 symbols
    test_symbols = symbols[:5]
    logging.info(f"Testing with {len(test_symbols)} symbols: {', '.join(test_symbols)}")
    
    # Process each symbol
    for i, symbol in enumerate(test_symbols, 1):
        logging.info(f"Processing {i}/{len(test_symbols)}: {symbol}")
        
        # Scrape data
        data = scrape_company_data(symbol)
        if data:
            # Store in database
            store_data_in_db(engine, symbol, data)
            logging.info(f"Successfully scraped and stored data for {symbol}")
        else:
            logging.error(f"Failed to scrape data for {symbol}")
        
        # Add delay to avoid overwhelming the server
        time.sleep(2)
    
    logging.info("Test scraping completed!")

if __name__ == "__main__":
    main() 