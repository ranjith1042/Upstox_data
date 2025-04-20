import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import time
from datetime import datetime
import re
import sqlite3
from sqlite3 import Error
import random

# Load environment variables
load_dotenv()

class ScreenerScraper:
    def __init__(self):
        self.base_url = "https://www.screener.in"
        self.session = requests.Session()
        self.email = os.getenv("SCREENER_EMAIL")
        self.password = os.getenv("SCREENER_PASSWORD")
        self.db_path = "financial_data.db"
        self.max_retries = 3
        self.base_delay = 5  # Base delay in seconds
        
        if not all([self.email, self.password]):
            raise ValueError("Please set SCREENER_EMAIL and SCREENER_PASSWORD in .env file")
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        
        # Initialize database connection
        self.conn = self.create_connection()

    def create_connection(self):
        """Create a database connection to the SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except Error as e:
            print(f"Error connecting to database: {e}")
            return None

    def create_table(self, table_name, df):
        """Create a table in the database from a DataFrame"""
        try:
            # Clean table name to be SQLite compatible
            clean_table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
            
            # Create table if it doesn't exist
            df.to_sql(clean_table_name, self.conn, if_exists='replace', index=False)
            print(f"Created/Updated table: {clean_table_name}")
            return True
        except Error as e:
            print(f"Error creating table {table_name}: {e}")
            return False

    def login(self):
        """Login to screener.in and establish a session"""
        try:
            # Get the login page first to extract CSRF token
            login_url = f"{self.base_url}/login"
            print(f"Accessing login page: {login_url}")
            
            # Create a new session
            self.session = requests.Session()
            
            # Set up session headers
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            })
            
            # Get the login page
            response = self.session.get(login_url)
            response.raise_for_status()
            
            # Extract CSRF token from the page
            soup = BeautifulSoup(response.text, 'html.parser')
            csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'})
            
            if not csrf_token:
                print("Could not find CSRF token on login page")
                print("Page content:", response.text[:500])
                return False
                
            csrf_token = csrf_token['value']
            print(f"Found CSRF token: {csrf_token[:10]}...")
            
            # Get the login form
            form = soup.find('form', {'method': 'post'})
            if not form:
                print("Could not find login form")
                return False
            
            # Get all form inputs
            form_data = {}
            for input_field in form.find_all('input'):
                name = input_field.get('name')
                if name:
                    form_data[name] = input_field.get('value', '')
            
            # Update credentials
            form_data.update({
                'username': self.email,
                'password': self.password
            })
            
            print(f"Form data keys: {list(form_data.keys())}")
            
            # Set headers for the login request
            login_headers = {
                'Origin': self.base_url,
                'Referer': login_url,
                'Content-Type': 'application/x-www-form-urlencoded',
                'Cache-Control': 'max-age=0',
                'Upgrade-Insecure-Requests': '1',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Sec-Fetch-Site': 'same-origin',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-User': '?1',
                'Sec-Fetch-Dest': 'document'
            }
            
            print(f"Attempting login with email: {self.email}")
            
            # Perform login
            login_response = self.session.post(
                login_url + "/",  # Add trailing slash
                data=form_data,
                headers=login_headers,
                allow_redirects=True
            )
            
            # Print response details for debugging
            print(f"Login response status: {login_response.status_code}")
            print(f"Login response URL: {login_response.url}")
            
            # Check if we were redirected to the home page
            if login_response.url == self.base_url + "/" or "dashboard" in login_response.url:
                print("Successfully logged in to screener.in")
                return True
            
            # Check if we're still on the login page
            if "Login - Screener" in login_response.text:
                print("Login failed - still on login page")
                # Try to find error message
                soup = BeautifulSoup(login_response.text, 'html.parser')
                error_msg = soup.find('div', {'class': 'error'}) or soup.find('ul', {'class': 'errorlist'})
                if error_msg:
                    print(f"Error message: {error_msg.text.strip()}")
                else:
                    # Look for any form-related errors
                    form_errors = soup.find_all(class_='error') or soup.find_all(class_='errorlist')
                    if form_errors:
                        print("Form errors found:")
                        for error in form_errors:
                            print(f"- {error.text.strip()}")
                    else:
                        print("No error message found. Response content:", login_response.text[:500])
                return False
            
            # If we got here, we're probably logged in
            print("Successfully logged in to screener.in")
            return True
            
        except Exception as e:
            print(f"Login failed: {str(e)}")
            return False

    def make_request(self, url, retry_count=0):
        """Make a request with exponential backoff for rate limiting"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and retry_count < self.max_retries:
                # Calculate delay with exponential backoff and jitter
                delay = self.base_delay * (2 ** retry_count) + random.uniform(0, 1)
                print(f"Rate limited. Waiting {delay:.2f} seconds before retry {retry_count + 1}/{self.max_retries}")
                time.sleep(delay)
                return self.make_request(url, retry_count + 1)
            else:
                raise

    def get_stock_data(self, symbol):
        """Get stock data for a given symbol"""
        try:
            # Clean up symbol (remove .NS if present)
            clean_symbol = symbol.replace('.NS', '')
            
            # Dictionary to store all data
            data = {
                'symbol': clean_symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'quarterly_pl': None,
                'annual_pl': None,
                'balance_sheet': None,
                'ratios': None
            }
            
            # 1. Get quarterly consolidated P/L
            quarterly_url = f"{self.base_url}/company/{clean_symbol}/consolidated/#quarters"
            print(f"\nFetching quarterly consolidated P/L from: {quarterly_url}")
            response = self.make_request(quarterly_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            data['quarterly_pl'] = self.extract_quarterly_results(soup)
            
            # Add random delay between requests
            time.sleep(random.uniform(3, 5))
            
            # 2. Get annual consolidated P/L
            annual_pl_url = f"{self.base_url}/company/{clean_symbol}/consolidated/#profit-loss"
            print(f"\nFetching annual consolidated P/L from: {annual_pl_url}")
            response = self.make_request(annual_pl_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            data['annual_pl'] = self.extract_profit_loss(soup)
            
            # Add random delay between requests
            time.sleep(random.uniform(3, 5))
            
            # 3. Get annual consolidated Balance Sheet
            bs_url = f"{self.base_url}/company/{clean_symbol}/consolidated/#balance-sheet"
            print(f"\nFetching annual consolidated Balance Sheet from: {bs_url}")
            response = self.make_request(bs_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            data['balance_sheet'] = self.extract_balance_sheet(soup)
            
            # Add random delay between requests
            time.sleep(random.uniform(3, 5))
            
            # 4. Get annual consolidated Ratios
            ratios_url = f"{self.base_url}/company/{clean_symbol}/consolidated/#ratios"
            print(f"\nFetching annual consolidated Ratios from: {ratios_url}")
            response = self.make_request(ratios_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            data['ratios'] = self.extract_ratios(soup)
            
            # Print data summary
            print(f"\nData summary for {symbol}:")
            print(f"- Quarterly P/L: {len(data['quarterly_pl'])} rows" if data['quarterly_pl'] is not None else "- Quarterly P/L: Not found")
            print(f"- Annual P/L: {len(data['annual_pl'])} rows" if data['annual_pl'] is not None else "- Annual P/L: Not found")
            print(f"- Balance Sheet: {len(data['balance_sheet'])} rows" if data['balance_sheet'] is not None else "- Balance Sheet: Not found")
            print(f"- Ratios: {len(data['ratios'])} rows" if data['ratios'] is not None else "- Ratios: Not found")
            
            # Store data in database
            self.store_data_in_db(clean_symbol, data)
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def store_data_in_db(self, symbol, data):
        """Store the extracted data in the database"""
        try:
            # Store quarterly P/L
            if data['quarterly_pl'] is not None and not data['quarterly_pl'].empty:
                table_name = f"IS_{symbol}_Q_C"  # Added _C suffix for consolidated
                self.create_table(table_name, data['quarterly_pl'])
                print(f"Saved quarterly consolidated P/L for {symbol}")
            
            # Store annual P/L
            if data['annual_pl'] is not None and not data['annual_pl'].empty:
                table_name = f"IS_{symbol}_A"
                self.create_table(table_name, data['annual_pl'])
                print(f"Saved annual P/L for {symbol}")
            
            # Store balance sheet
            if data['balance_sheet'] is not None and not data['balance_sheet'].empty:
                table_name = f"BS_{symbol}_A"
                self.create_table(table_name, data['balance_sheet'])
                print(f"Saved balance sheet for {symbol}")
            
            # Store ratios
            if data['ratios'] is not None and not data['ratios'].empty:
                table_name = f"RATIOS_{symbol}"
                self.create_table(table_name, data['ratios'])
                print(f"Saved ratios for {symbol}")
            
        except Exception as e:
            print(f"Error storing data in database: {e}")

    def extract_ratios(self, soup):
        """Extract key ratios from the page."""
        try:
            # Find the ratios section
            ratios_section = soup.find('section', {'id': 'ratios'})
            if not ratios_section:
                print("Could not find ratios section")
                return pd.DataFrame()
            
            # Find all ratio rows
            ratio_rows = ratios_section.find_all('tr')
            if not ratio_rows:
                print("No ratio rows found")
                return pd.DataFrame()
            
            # Extract ratio data
            ratios = []
            for row in ratio_rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    ratio_name = cells[0].text.strip()
                    ratio_value = cells[1].text.strip()
                    ratios.append({'Ratio': ratio_name, 'Value': ratio_value})
            
            if not ratios:
                print("No ratios data found")
                return pd.DataFrame()
            
            df = pd.DataFrame(ratios)
            print(f"Extracted {len(df)} ratios")
            return df
        except Exception as e:
            print(f"Error extracting ratios: {str(e)}")
            return pd.DataFrame()

    def extract_quarterly_results(self, soup):
        """Extract quarterly results data from the page."""
        try:
            # Find the quarterly results table - try different selectors
            table = (
                soup.find('table', {'id': 'quarters'}) or 
                soup.find('table', {'class': 'data-table responsive-text-nowrap'}) or
                soup.find('section', {'id': 'quarters'}).find('table') if soup.find('section', {'id': 'quarters'}) else None
            )
            
            if not table:
                print("Could not find quarterly results table")
                return pd.DataFrame()
            
            # Print table HTML for debugging
            print("Found quarterly results table:")
            print(str(table)[:200] + "...")
            
            # Extract headers
            headers = ['Metric']  # First column is for metric names
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td'])[1:]:  # Skip first column
                    header_text = th.text.strip()
                    if header_text:
                        headers.append(header_text)
            
            if len(headers) <= 1:
                print("No headers found in quarterly results table")
                return pd.DataFrame()
            
            print(f"Found headers: {headers}")
            
            # Extract data rows
            rows = []
            data_rows = table.find_all('tr')[1:]  # Skip header row
            for tr in data_rows:
                cells = tr.find_all(['td', 'th'])
                if cells:
                    row_data = []
                    metric_name = cells[0].text.strip()  # First cell is metric name
                    row_data.append(metric_name)
                    
                    # Add values for each period
                    for td in cells[1:]:
                        value = td.text.strip()
                        row_data.append(value)
                    
                    if len(row_data) == len(headers):  # Only add complete rows
                        rows.append(row_data)
            
            if not rows:
                print("No quarterly results data found")
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=headers)
            print(f"Extracted {len(df)} quarterly results rows with {len(df.columns)} columns")
            print("Sample data:")
            print(df.head())
            return df
            
        except Exception as e:
            print(f"Error extracting quarterly results: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def extract_profit_loss(self, soup):
        """Extract profit & loss data from the page."""
        try:
            # Find the profit & loss table - try different selectors
            table = (
                soup.find('table', {'id': 'profit-loss'}) or 
                soup.find('section', {'id': 'profit-loss'}).find('table') if soup.find('section', {'id': 'profit-loss'}) else None
            )
            
            if not table:
                print("Could not find profit & loss table")
                return pd.DataFrame()
            
            # Extract headers
            headers = ['Metric']  # First column is for metric names
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td'])[1:]:  # Skip first column
                    header_text = th.text.strip()
                    if header_text:
                        headers.append(header_text)
            
            if len(headers) <= 1:
                print("No headers found in profit & loss table")
                return pd.DataFrame()
            
            # Extract data rows
            rows = []
            data_rows = table.find_all('tr')[1:]  # Skip header row
            for tr in data_rows:
                cells = tr.find_all(['td', 'th'])
                if cells:
                    row_data = []
                    metric_name = cells[0].text.strip()  # First cell is metric name
                    row_data.append(metric_name)
                    
                    # Add values for each period
                    for td in cells[1:]:
                        value = td.text.strip()
                        row_data.append(value)
                    
                    if len(row_data) == len(headers):  # Only add complete rows
                        rows.append(row_data)
            
            if not rows:
                print("No profit & loss data found")
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=headers)
            print(f"Extracted {len(df)} profit & loss rows with {len(df.columns)} columns")
            print("Sample data:")
            print(df.head())
            return df
            
        except Exception as e:
            print(f"Error extracting profit & loss: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def extract_balance_sheet(self, soup):
        """Extract balance sheet data from the page."""
        try:
            # Find the balance sheet table - try different selectors
            table = (
                soup.find('table', {'id': 'balance-sheet'}) or 
                soup.find('section', {'id': 'balance-sheet'}).find('table') if soup.find('section', {'id': 'balance-sheet'}) else None
            )
            
            if not table:
                print("Could not find balance sheet table")
                return pd.DataFrame()
            
            # Extract headers
            headers = ['Metric']  # First column is for metric names
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td'])[1:]:  # Skip first column
                    header_text = th.text.strip()
                    if header_text:
                        headers.append(header_text)
            
            if len(headers) <= 1:
                print("No headers found in balance sheet table")
                return pd.DataFrame()
            
            # Extract data rows
            rows = []
            data_rows = table.find_all('tr')[1:]  # Skip header row
            for tr in data_rows:
                cells = tr.find_all(['td', 'th'])
                if cells:
                    row_data = []
                    metric_name = cells[0].text.strip()  # First cell is metric name
                    row_data.append(metric_name)
                    
                    # Add values for each period
                    for td in cells[1:]:
                        value = td.text.strip()
                        row_data.append(value)
                    
                    if len(row_data) == len(headers):  # Only add complete rows
                        rows.append(row_data)
            
            if not rows:
                print("No balance sheet data found")
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=headers)
            print(f"Extracted {len(df)} balance sheet rows with {len(df.columns)} columns")
            print("Sample data:")
            print(df.head())
            return df
            
        except Exception as e:
            print(f"Error extracting balance sheet: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def extract_cash_flows(self, soup):
        """Extract cash flow data from the page."""
        try:
            # Find the cash flow table - try different selectors
            table = (
                soup.find('table', {'id': 'cash-flow'}) or 
                soup.find('section', {'id': 'cash-flow'}).find('table') if soup.find('section', {'id': 'cash-flow'}) else None
            )
            
            if not table:
                print("Could not find cash flow table")
                return pd.DataFrame()
            
            # Extract headers
            headers = ['Metric']  # First column is for metric names
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td'])[1:]:  # Skip first column
                    header_text = th.text.strip()
                    if header_text:
                        headers.append(header_text)
            
            if len(headers) <= 1:
                print("No headers found in cash flow table")
                return pd.DataFrame()
            
            # Extract data rows
            rows = []
            data_rows = table.find_all('tr')[1:]  # Skip header row
            for tr in data_rows:
                cells = tr.find_all(['td', 'th'])
                if cells:
                    row_data = []
                    metric_name = cells[0].text.strip()  # First cell is metric name
                    row_data.append(metric_name)
                    
                    # Add values for each period
                    for td in cells[1:]:
                        value = td.text.strip()
                        row_data.append(value)
                    
                    if len(row_data) == len(headers):  # Only add complete rows
                        rows.append(row_data)
            
            if not rows:
                print("No cash flow data found")
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=headers)
            print(f"Extracted {len(df)} cash flow rows with {len(df.columns)} columns")
            print("Sample data:")
            print(df.head())
            return df
            
        except Exception as e:
            print(f"Error extracting cash flows: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

def get_symbols_from_file(start_symbol=None):
    """Read symbols from symbols.tls file, optionally starting from a specific symbol"""
    try:
        with open('symbols.tls', 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        
        if start_symbol:
            try:
                start_index = symbols.index(start_symbol)
                return symbols[start_index:]
            except ValueError:
                print(f"Start symbol {start_symbol} not found in symbols.tls")
                return symbols
        return symbols
    except FileNotFoundError:
        print("symbols.tls file not found")
        return []

def main():
    try:
        # Initialize scraper
        scraper = ScreenerScraper()
        
        # Login to screener.in
        if not scraper.login():
            print("Failed to login. Exiting...")
            return
        
        # Get symbols from file, starting from BEML
        start_symbol = "BEML.NS"
        symbols = get_symbols_from_file(start_symbol)
        if not symbols:
            print("No symbols found in symbols.tls")
            return
        
        total_symbols = len(symbols)
        print(f"\nProcessing {total_symbols} symbols from symbols.tls, starting from {start_symbol}")
        
        # Process each symbol
        for i, symbol in enumerate(symbols, 1):
            print(f"\nProcessing {i}/{total_symbols}: {symbol}...")
            
            # Get data for the symbol
            data = scraper.get_stock_data(symbol)
            
            # Add longer delay between symbols
            if i < total_symbols:  # Don't wait after the last symbol
                delay = random.uniform(10, 15)
                print(f"\nWaiting {delay:.2f} seconds before processing next symbol...")
                time.sleep(delay)
        
        print("\nData scraping completed!")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Close database connection
        if hasattr(scraper, 'conn'):
            scraper.conn.close()

if __name__ == "__main__":
    main() 