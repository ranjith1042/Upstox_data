import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
from scipy.stats import linregress
import multiprocessing as mp
from functools import partial
import gc

def get_stock_data_from_db(symbol: str) -> pd.DataFrame:
    """Fetch stock data from SQLite database"""
    try:
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        
        # Clean symbol name and remove .NS suffix if present
        clean_symbol = symbol.replace('.NS', '').replace('&', 'AND').replace('-', '_').replace(' ', '_')
        
        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{clean_symbol}'")
        if not cursor.fetchone():
            # Try uppercase version
            alt_symbol = clean_symbol.upper()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{alt_symbol}'")
            if not cursor.fetchone():
                conn.close()
                return None
            clean_symbol = alt_symbol
            
        # Execute query with proper date handling
        query = f"""
        SELECT 
            timestamp as date,
            open, high, low, close, volume 
        FROM "{clean_symbol}"
        ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            conn.close()
            return None
            
        # Convert date to datetime and ensure it's timezone-naive
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Verify required columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            conn.close()
            return None
            
        conn.close()
        return df
        
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        return None

def calculate_momentum(data: pd.DataFrame) -> float:
    """Calculate momentum using vectorized operations"""
    try:
        lookback_days = 252
        
        if len(data) < lookback_days:
            return None
            
        # Get last lookback_days of data
        prices = data['close'].iloc[-lookback_days:].values
        log_prices = np.log(prices)
        
        # Vectorized linear regression
        x = np.arange(len(log_prices))
        x_mean = x.mean()
        y_mean = log_prices.mean()
        
        # Calculate slope using vectorized operations
        numerator = np.sum((x - x_mean) * (log_prices - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        slope = numerator / denominator
        
        # Calculate R-squared
        y_pred = slope * (x - x_mean) + y_mean
        r2 = 1 - np.sum((log_prices - y_pred) ** 2) / np.sum((log_prices - y_mean) ** 2)
        
        # Calculate volatility
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Calculate coefficient of variation
        lookback_cv = np.std(prices) / np.mean(prices)
        
        # Calculate momentum score
        momentum = (np.exp(slope * 250) - 1) * r2 / (1 + lookback_cv)
        
        return momentum if np.isfinite(momentum) else None
        
    except Exception:
            return None
            
def calculate_obv_trend(df):
    """Calculate OBV trend using vectorized operations"""
    try:
        if len(df) < 252:
            return None
            
        # Create a copy and ensure float64 dtype for all numeric columns
        df_copy = df.copy()
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype('float64')
        
        # Calculate price changes
        price_changes = df_copy['close'].diff()
        
        # Handle zero volumes using vectorized operations
        rolling_mean = df_copy['volume'].rolling(window=20, min_periods=1).mean()
        zero_mask = df_copy['volume'] == 0
        df_copy.loc[zero_mask, 'volume'] = rolling_mean[zero_mask]
        
        # Forward and backward fill any remaining zeros
        df_copy['volume'] = df_copy['volume'].replace(0, np.nan).ffill().bfill()
        
        # Calculate OBV using vectorized operations
        df_copy['obv'] = (np.sign(price_changes) * df_copy['volume']).cumsum()
        
        # Get last 12 months of data
        last_12m_obv = df_copy['obv'].iloc[-252:].values
        
        # Vectorized linear regression
        x = np.arange(len(last_12m_obv))
        x_mean = x.mean()
        y_mean = last_12m_obv.mean()
    
        # Calculate slope
        numerator = np.sum((x - x_mean) * (last_12m_obv - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        slope = numerator / denominator
        
        # Calculate standard deviation
        std_dev = np.std(last_12m_obv)
        if std_dev <= 1e-6:
            return 0.0
            
        # Normalize trend strength
        trend_strength = np.tanh(slope / std_dev)
    
        return trend_strength if np.isfinite(trend_strength) else None
        
    except Exception:
        return None
        
def calculate_away_from_52w_high(data: pd.DataFrame) -> float:
    """Calculate how far current price is from 52-week high"""
    if len(data) < 252:
        return None
    try:
        high_52w = data['high'].iloc[-252:].max()
        current_price = data['close'].iloc[-1]
        if high_52w <= 0 or not np.isfinite(high_52w) or not np.isfinite(current_price):
            return None
        return (current_price - high_52w) / high_52w
    except Exception as e:
        print(f"Error calculating away from 52w high: {str(e)}")
        return None

def calculate_above_ma(data: pd.DataFrame) -> float:
    """Calculate how far current price is above 200-day moving average"""
    if len(data) < 200:
        return None
    try:
        ma_200 = data['close'].iloc[-200:].mean()
        current_price = data['close'].iloc[-1]
        if ma_200 <= 0 or not np.isfinite(ma_200) or not np.isfinite(current_price):
            return None
        return (current_price - ma_200) / ma_200
    except Exception as e:
        print(f"Error calculating above MA: {str(e)}")
        return None

def prepare_training_data(symbols, start_date, end_date):
    """Prepare training data for all symbols with improved data quality checks"""
    X = []
    y = []
    valid_symbols = []
    
    total_symbols = len(symbols)
    processed = 0
    
    for symbol in symbols:
        try:
            processed += 1
            print(f"\nProcessing {symbol} ({processed}/{total_symbols})")
            
            # Get stock data
            df = get_stock_data_from_db(symbol)
            if df is None:
                print(f"  ❌ No stock data found for {symbol}")
                continue
                
            # Check for minimum data requirements
            if len(df) < 504:  # Need at least 2 years of data
                print(f"  ❌ Insufficient stock data points for {symbol} (need 504, found {len(df)})")
                print(f"    Available data from {df['date'].min()} to {df['date'].max()}")
                continue
            
            # Create a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Add detailed date range logging
            print(f"\n  📅 Detailed date range information for {symbol}:")
            print(f"    - Data available from: {df['date'].min()}")
            print(f"    - Data available until: {df['date'].max()}")
            print(f"    - Total data points: {len(df)}")
            
            # Filter data for training period
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            if len(df) < 504:  # Need at least 2 years of data
                print(f"  ❌ Insufficient data points in training period for {symbol} (need 504, found {len(df)})")
                print(f"    Available data from {df['date'].min()} to {df['date'].max()}")
                continue
            
            # Convert dates to datetime if they're not already
            df['date'] = pd.to_datetime(df['date'])
            
            # Get monthly dates for processing
            monthly_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='M')
            print(f"\n  📅 Processing months from {df['date'].min()} to {df['date'].max()}")
            print(f"  📊 Total data points: {len(df)}")
            print(f"  🔄 Monthly dates to process: {monthly_dates}")
            
            symbol_samples = 0  # Counter for samples from this symbol
            
            for month_date in monthly_dates:
                print(f"\n  Processing month {month_date}:")
                
                # Skip if we don't have enough data after this month
                if month_date + pd.Timedelta(days=365) > df['date'].max():
                    print(f"    ⚠️ Skipping - need data until {month_date + pd.Timedelta(days=365)}, but only have until {df['date'].max()}")
                    continue
                
                # Get data up to current month for features
                feature_data = df[df['date'] <= month_date]
                if len(feature_data) < 252:  # Need at least 1 year of data for features
                    print(f"    ⚠️ Skipping - insufficient feature data (need 252, found {len(feature_data)})")
                    print(f"      Feature data from {feature_data['date'].min()} to {feature_data['date'].max()}")
                    continue
                
                # Get data for return calculation
                return_data = df[(df['date'] > month_date) & (df['date'] <= df['date'].max())]
                if len(return_data) < 252:  # Need at least 252 trading days (1 year) for returns
                    print(f"    ⚠️ Skipping - insufficient return data (need 252, found {len(return_data)})")
                    print(f"      Return data from {return_data['date'].min() if not return_data.empty else 'N/A'} to {return_data['date'].max() if not return_data.empty else 'N/A'}")
                    continue
                
                print(f"    ✅ Data requirements met:")
                print(f"      - Feature data: {len(feature_data)} points from {feature_data['date'].min()} to {feature_data['date'].max()}")
                print(f"      - Return data: {len(return_data)} points from {return_data['date'].min()} to {return_data['date'].max()}")
                
                # Calculate features using data up to current month
                momentum_score = calculate_momentum(feature_data)
                obv_trend = calculate_obv_trend(feature_data)
                away_from_high = calculate_away_from_52w_high(feature_data)
                above_ma = calculate_above_ma(feature_data)
                
                # Check each feature
                if momentum_score is None or pd.isna(momentum_score):
                    print(f"  ❌ Invalid momentum_score for month {month_date}")
                    continue
                if obv_trend is None or pd.isna(obv_trend):
                    print(f"  ❌ Invalid obv_trend for month {month_date}")
                    continue
                if away_from_high is None or pd.isna(away_from_high):
                    print(f"  ❌ Invalid away_from_high for month {month_date}")
                    continue
                if above_ma is None or pd.isna(above_ma):
                    print(f"  ❌ Invalid above_ma for month {month_date}")
                    continue
                
                # Get current price and price one year ahead
                current_price = feature_data['close'].iloc[-1]
                future_price = return_data['close'].iloc[-1]
                
                # Skip if prices are invalid
                if current_price <= 0 or future_price <= 0:
                    print(f"  ❌ Invalid prices for month {month_date}")
                    continue
                    
                # Calculate return
                one_year_return = (future_price - current_price) / current_price
                
                # Skip if return is NaN or infinite
                if not np.isfinite(one_year_return):
                    print(f"  ❌ Invalid return calculation for month {month_date}")
                    continue
                
                # Add to training data
                X.append([momentum_score, obv_trend, away_from_high, above_ma])
                y.append(one_year_return)
                valid_symbols.append(symbol)
                symbol_samples += 1
                print(f"  ✅ Added training sample for month {month_date}")
            
            if symbol_samples > 0:
                print(f"  ✅ {symbol}: {symbol_samples} training samples")
            else:
                print(f"  ❌ {symbol}: No valid training samples found")
            
        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {str(e)}")
            continue
    
    if not X or not y:
        print("No valid training data found!")
        return None, None, None
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Create a mask for valid values
    valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    valid_symbols = [symbol for i, symbol in enumerate(valid_symbols) if valid_mask[i]]
    
    if len(X) == 0 or len(y) == 0:
        print("No valid training data after filtering!")
        return None, None, None
        
    print(f"\nFound {len(valid_symbols)} valid training samples")
    return X, y, valid_symbols

def calculate_cagr(start_price, end_price, years):
    """Calculate Compound Annual Growth Rate (CAGR)"""
    if start_price <= 0 or years <= 0:
        return 0
    return (end_price / start_price) ** (1 / years) - 1

def calculate_max_drawdown(prices):
    """Calculate Maximum Drawdown from price series"""
    try:
        if len(prices) < 2:
            return 0
            
        # Convert to numpy array and remove any zero or negative prices
        prices = np.array(prices)
        valid_mask = (prices > 0) & np.isfinite(prices)
        prices = prices[valid_mask]
        
        if len(prices) < 2:
            return 0
            
        # Calculate daily returns
        returns = np.diff(prices) / prices[:-1]
        
        # Handle any invalid returns
        returns = returns[np.isfinite(returns)]
        
        if len(returns) == 0:
            return 0
            
        # Convert to cumulative returns
        cumulative = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdowns
        drawdowns = (cumulative - running_max) / running_max
        
        return float(np.min(drawdowns))
    except Exception as e:
        print(f"Error calculating max drawdown: {str(e)}")
        return 0

def get_all_symbols_data():
    """Get all symbols and their date ranges in one database query"""
    conn = sqlite3.connect('stock_data.db')
    try:
        # Get all symbols from sqlite_master
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
        symbols = [row[0] for row in cursor.fetchall()]
        
        # Get date ranges for each symbol
        date_ranges = []
        for symbol in symbols:
            try:
                query = f"""
                SELECT 
                    DATE(MIN(timestamp)) as earliest_date,
                    DATE(MAX(timestamp)) as latest_date
                FROM "{symbol}"
                """
                dates = pd.read_sql_query(query, conn)
                if not dates.empty and not dates['earliest_date'].isna().any() and not dates['latest_date'].isna().any():
                    date_ranges.append({
                        'symbol': symbol,
                        'earliest_date': dates['earliest_date'].iloc[0],
                        'latest_date': dates['latest_date'].iloc[0]
                    })
            except Exception as e:
                print(f"Error getting dates for {symbol}: {str(e)}")
                continue
                
        return pd.DataFrame(date_ranges)
    finally:
        conn.close()

def train_xgboost_model(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple:
    """Train XGBoost model for a given symbol and date range."""
    try:
        # Get all data for the symbol
        df = get_stock_data_from_db(symbol)
        
        if df is None or len(df) < 252:  # Need at least 1 year of data
            return None, None, None
        
        # Pre-allocate arrays for better performance
        n_samples = len(df) - 252
        all_features = np.zeros((n_samples, 4))  # 4 features
        all_labels = np.zeros(n_samples)
        dates = []
        prices = []
        
        # Calculate features for each day - vectorized operations where possible
        valid_samples = 0
        
        # Pre-calculate price changes and moving averages
        df['price_change'] = df['close'].diff()
        df['ma_200'] = df['close'].rolling(window=200).mean()
        df['high_52w'] = df['high'].rolling(window=252).max()
        
        for i in range(252, len(df)):
            try:
                # Get data up to current day for features
                feature_data = df.iloc[:i]
                
                # Calculate features using pre-calculated values
                momentum = (feature_data['close'].iloc[-1] - feature_data['close'].iloc[-252]) / feature_data['close'].iloc[-252]
                obv_trend = calculate_obv_trend(feature_data)
                away_from_high = (feature_data['close'].iloc[-1] - feature_data['high_52w'].iloc[-1]) / feature_data['high_52w'].iloc[-1]
                above_ma = (feature_data['close'].iloc[-1] - feature_data['ma_200'].iloc[-1]) / feature_data['ma_200'].iloc[-1]
                
                # Check each feature
                if any(not np.isfinite(x) for x in [momentum, obv_trend, away_from_high, above_ma]):
                    continue
                    
                # Calculate forward return (label) - next 252 days
                if i + 252 < len(df):
                    start_price = df.iloc[i]['close']
                    end_price = df.iloc[i + 252]['close']
                    forward_return = (end_price - start_price) / start_price
                    
                    if np.isfinite(forward_return):
                        # Store features and labels
                        all_features[valid_samples] = [momentum, obv_trend, away_from_high, above_ma]
                        all_labels[valid_samples] = forward_return
                        dates.append(df['date'].iloc[i])
                        prices.append(start_price)
                        valid_samples += 1
            except Exception:
                continue
                
        # Trim arrays to actual size
        all_features = all_features[:valid_samples]
        all_labels = all_labels[:valid_samples]
        
        if valid_samples == 0:
            return None, None, None
            
        # Convert to numpy arrays
        X = all_features
        y = all_labels
        dates = np.array(dates)
        prices = np.array(prices)
        
        # Split into train and validation sets (80-20 split based on time)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        dates_val = dates[split_idx:]
        prices_val = prices[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Clip extreme returns
        y_train = np.clip(y_train, -1, 1)
        y_val = np.clip(y_val, -1, 1)
        
        # Create and train model with optimized parameters for speed
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',  # Use histogram-based algorithm
            n_estimators=50,     # Reduced number of trees
            max_depth=3,         # Reduced tree depth
            learning_rate=0.2,   # Increased learning rate
            subsample=0.8,       # Reduced subsample ratio
            colsample_bytree=0.8,# Reduced column sample ratio
            min_child_weight=2,  # Reduced min_child_weight
            gamma=0.2,           # Increased gamma
            n_jobs=-1,           # Use all available cores
            early_stopping_rounds=3,  # Reduced early stopping rounds
            max_bin=256,         # Reduced number of bins for histogram
            grow_policy='lossguide'  # Use loss-guided growth
        )
        
        model.fit(
            X_train_scaled, 
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=0
        )
        
        # Make predictions
        val_predictions = model.predict(X_val_scaled)
        
        return model, (dates_val, prices_val, y_val, val_predictions), scaler
        
    except Exception as e:
        print(f"Error in train_xgboost_model: {str(e)}")
        return None, None, None
        
def predict_next_year(symbol: str, model, scaler, prediction_start_date: pd.Timestamp):
    """Make prediction for the next year (252 trading days)."""
    try:
        # Get historical data
        df = get_stock_data_from_db(symbol)
        
        if df is None or len(df) < 252:
            print(f"Insufficient historical data for prediction")
            return None
            
        # Calculate features
        momentum = calculate_momentum(df)
        obv_trend = calculate_obv_trend(df)
        away_from_high = calculate_away_from_52w_high(df)
        above_ma = calculate_above_ma(df)
        
        # Skip if any feature is None or NaN
        if any(x is None for x in [momentum, obv_trend, away_from_high, above_ma]):
            return None
            
        # Create feature vector as a 1D array
        features = np.array([
            momentum,
            obv_trend,
            away_from_high,
            above_ma
        ])
        
        # Reshape to 2D array for scaler
        features = features.reshape(1, -1)
        
        # Scale features using the provided scaler
        if isinstance(scaler, tuple):
            scaler = scaler[0]  # Extract scaler from tuple if needed
            
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return prediction
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None

def get_stock_data(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Get stock data from the database for a given symbol and date range."""
    try:
        # Clean the symbol name to handle special characters
        clean_symbol = symbol.replace('&', '_').replace('-', '_')
        
        # Connect to the database
        conn = sqlite3.connect('stock_data.db')
        
        # Check if the table exists
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{clean_symbol}'")
        if not cursor.fetchone():
            print(f"No data table found for {symbol}")
            conn.close()
            return None
            
        # Get data for the date range using DATE() function to extract date from timestamp
        query = f"""
        SELECT 
            DATE(timestamp) as date,
            open, high, low, close, volume
        FROM "{clean_symbol}"
        WHERE DATE(timestamp) BETWEEN ? AND ?
        GROUP BY DATE(timestamp)  -- Group by date to get one record per day
        ORDER BY date
        """
        
        df = pd.read_sql_query(
            query,
            conn,
            params=(start_date.strftime('%Y-%m-%d'),
                   end_date.strftime('%Y-%m-%d'))
        )
        
        conn.close()
        
        if df.empty:
            print(f"No data found for {symbol} in the specified date range")
            return None
            
        # Convert date column to datetime.date and set as index
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.set_index('date', inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def process_symbol(args):
    """Process a single symbol with its date range"""
    symbol, symbol_earliest, symbol_latest, prediction_end = args
    
    try:
        # Set training period for this symbol
        training_end = prediction_end
        training_start = symbol_earliest
        
        # Skip if we don't have enough training data
        if (training_end - training_start).days < 365:
            return None
            
        # Train model for this symbol
        model, val_data, scaler = train_xgboost_model(symbol, training_start, training_end)
        
        if model is None or val_data is None:
            return None
            
        # Make prediction for next year
        prediction = predict_next_year(symbol, model, scaler, prediction_end)
        
        if prediction is None:
            return None
            
        # Get current price
        df = get_stock_data_from_db(symbol)
        if df is None:
            return None
            
        current_price = df['close'].iloc[-1]
        
        return {
            'symbol': symbol,
            'prediction': prediction,
            'current_price': current_price,
            'model': model,
            'scaler': scaler,
            'validation_data': val_data
        }
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None

def calculate_portfolio_value(positions, cash, current_prices):
    """Calculate total portfolio value including cash"""
    total_value = cash  # Start with cash
    for symbol, position_info in positions.items():
        if symbol in current_prices:
            quantity = position_info['quantity']  # Get quantity from position info
            price = current_prices[symbol]
            position_value = quantity * price
            total_value += position_value
    return total_value

def calculate_position_size(portfolio_value, price, is_new_position=True):
    """Calculate position size with risk management
    - New positions start at 5% of portfolio
    - Roll-ups add 5% increments
    - Maximum position size is 20% of portfolio
    """
    if is_new_position:
        target_position_value = portfolio_value * 0.05  # 5% for new positions
    else:
        target_position_value = portfolio_value * 0.05  # 5% for roll-ups
    
    return int(target_position_value / price)  # Round down to whole shares

def rebalance_portfolio(current_positions, target_positions, prices, cash):
    """Rebalance portfolio to match target positions"""
    trades = []
    new_cash = cash
    
    # First sell positions not in target
    for symbol in current_positions:
        if symbol not in target_positions:
            quantity = current_positions[symbol]
            trade_value = quantity * prices[symbol]
            new_cash += trade_value
            trades.append(('SELL', symbol, quantity))
        
    # Calculate target position value
    total_value = calculate_portfolio_value(current_positions, new_cash, prices)
    target_position_value = total_value / len(target_positions)
    
    # Buy new positions or roll up existing ones
    for symbol in target_positions:
        current_quantity = current_positions.get(symbol, 0)
        target_quantity = int(target_position_value / prices[symbol])
        
        if target_quantity > current_quantity:
            # Buy more
            buy_quantity = target_quantity - current_quantity
            cost = buy_quantity * prices[symbol]
            if cost <= new_cash:
                new_cash -= cost
                trades.append(('BUY', symbol, buy_quantity))
        elif target_quantity < current_quantity:
            # Sell excess
            sell_quantity = current_quantity - target_quantity
            new_cash += sell_quantity * prices[symbol]
            trades.append(('SELL', symbol, sell_quantity))
            
    return trades, new_cash

def select_stocks(predictions, prices, min_stocks=0, max_stocks=30):
    """Select stocks based on predictions with minimum requirements"""
    if len(predictions) == 0:
        return []
        
    # Sort stocks by predicted return
    sorted_stocks = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 10 stocks, but no more than max_stocks
    num_stocks = min(max_stocks, 10)
    
    # Filter for minimum price and volume requirements
    selected_stocks = []
    for symbol, pred_return in sorted_stocks:
        if len(selected_stocks) >= num_stocks:
            break
            
        price = prices.get(symbol)
        if price:  # Only check if price exists, no minimum price requirement
            selected_stocks.append(symbol)
            
    return selected_stocks

def standardize_date(date):
    """Standardize date format and handling"""
    if isinstance(date, str):
        return pd.to_datetime(date).date()
    elif isinstance(date, pd.Timestamp):
        return date.date()
    elif isinstance(date, datetime.date):
        return date
    else:
        raise ValueError(f"Unsupported date format: {type(date)}")

def preprocess_and_cache_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess and cache data for faster access"""
    try:
        # Create a copy to avoid modifying original
        df_cache = df.copy()
        
        # Convert date to datetime if not already
        df_cache['date'] = pd.to_datetime(df_cache['date'])
        
        # Convert volume to float64 at the start
        df_cache['volume'] = df_cache['volume'].astype('float64')
        
        # Pre-calculate commonly used values
        df_cache['price_change'] = df_cache['close'].diff()
        df_cache['returns'] = df_cache['close'].pct_change()
        
        # Pre-calculate moving averages
        df_cache['ma_200'] = df_cache['close'].rolling(window=200, min_periods=1).mean()
        df_cache['ma_50'] = df_cache['close'].rolling(window=50, min_periods=1).mean()
        
        # Pre-calculate highs and lows
        df_cache['high_52w'] = df_cache['high'].rolling(window=252, min_periods=1).max()
        df_cache['low_52w'] = df_cache['low'].rolling(window=252, min_periods=1).min()
        
        # Pre-calculate volume metrics
        df_cache['volume_ma_20'] = df_cache['volume'].rolling(window=20, min_periods=1).mean()
        df_cache['volume_ma_50'] = df_cache['volume'].rolling(window=50, min_periods=1).mean()
        
        # Handle zero volumes - all operations now use float64
        zero_volume_mask = df_cache['volume'] == 0
        df_cache.loc[zero_volume_mask, 'volume'] = df_cache.loc[zero_volume_mask, 'volume_ma_20']
        
        # Calculate OBV
        df_cache['obv'] = (np.sign(df_cache['price_change']) * df_cache['volume']).cumsum()
        
        # Set date as index for faster lookups
        df_cache.set_index('date', inplace=True)
        
        return df_cache
        
    except Exception as e:
        print(f"Error in preprocess_and_cache_data: {str(e)}")
        return df

def batch_fetch_stock_data(symbols: list) -> dict:
    """Fetch stock data for multiple symbols in a single database connection"""
    try:
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        
        # Get all available tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        available_tables = {table[0] for table in cursor.fetchall()}
        
        # Clean symbols and check availability
        clean_symbols = {}
        for symbol in symbols:
            clean_symbol = symbol.replace('.NS', '').replace('&', 'AND').replace('-', '_').replace(' ', '_')
            if clean_symbol in available_tables:
                clean_symbols[symbol] = clean_symbol
            elif clean_symbol.upper() in available_tables:
                clean_symbols[symbol] = clean_symbol.upper()
        
        # Batch fetch data for all valid symbols
        data_dict = {}
        for symbol, clean_symbol in clean_symbols.items():
            query = f"""
            SELECT 
                timestamp as date,
                open, high, low, close, volume 
            FROM "{clean_symbol}"
            ORDER BY timestamp
            """
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                # Preprocess and cache the data
                df = preprocess_and_cache_data(df)
                data_dict[symbol] = df
        
        conn.close()
        return data_dict
        
    except Exception:
        if 'conn' in locals():
            conn.close()
        return {}

def main():
    """Main function to run the model"""
    try:
        # Initialize trade history list
        trade_history = []
        
        # Initialize feature importance tracking
        feature_names = ['Momentum', 'OBV Trend', 'Away from 52w High', 'Above MA']
        all_importance_scores = []
        all_feature_values = []
        
        # Get all symbols and their date ranges in one query
        symbols_df = get_all_symbols_data()
        if symbols_df.empty:
            print("No symbols found in the database!")
            return
            
        # Take only first 50 symbols
        symbols_df = symbols_df.head(50)
        print(f"\nProcessing {len(symbols_df)} symbols")
        
        # Batch fetch all stock data at once
        stock_data_cache = batch_fetch_stock_data(symbols_df['symbol'].tolist())
        
        # Convert dates to datetime for comparison
        symbols_df['latest_date'] = pd.to_datetime(symbols_df['latest_date']).dt.date
        symbols_df['earliest_date'] = pd.to_datetime(symbols_df['earliest_date']).dt.date
        
        # Find the latest date for prediction
        latest_date = max(symbols_df['latest_date'])
        prediction_end = latest_date
        prediction_start = min(symbols_df['earliest_date'])
        
        # Filter valid symbols
        valid_symbols = symbols_df[
            (symbols_df['latest_date'] >= prediction_start) &
            (symbols_df['earliest_date'] <= prediction_end)
        ]
        
        if valid_symbols.empty:
            print("No valid data found!")
            return
        
        # Prepare arguments for parallel processing
        process_args = [
            (row['symbol'], 
             row['earliest_date'],
             row['latest_date'],
             prediction_end)
            for _, row in valid_symbols.iterrows()
        ]
        
        # Process symbols in parallel with optimized chunk size
        num_cores = mp.cpu_count()
        chunk_size = max(1, len(process_args) // (num_cores * 2))
        all_val_data = []
        
        print(f"\nProcessing using {num_cores} cores")
        
        # Use context manager for better resource management
        with mp.Pool(num_cores) as pool:
            for i in range(0, len(process_args), chunk_size):
                chunk = process_args[i:i + chunk_size]
                
                # Process chunk with progress bar
                chunk_results = list(tqdm(
                    pool.imap(process_symbol, chunk),
                    total=len(chunk),
                    desc=f"Chunk {i//chunk_size + 1}"
                ))
                
                # Filter and extend results
                successful_results = [r for r in chunk_results if r is not None]
                all_val_data.extend(successful_results)
                
                # Collect feature importance scores and feature values
                for result in successful_results:
                    if result and 'model' in result:
                        importance = result['model'].feature_importances_
                        all_importance_scores.append(importance)
                        
                        # Get feature values from validation data
                        dates_val, prices_val, y_val, val_predictions = result['validation_data']
                        feature_values = result['model'].get_booster().get_score(importance_type='gain')
                        all_feature_values.append(feature_values)
                
                # Clear memory after each chunk
                del chunk_results
                gc.collect()
        
        if not all_val_data:
            print("No valid data found after processing!")
            return
        
        # Calculate average feature importance across all models
        if all_importance_scores:
            avg_importance = np.mean(all_importance_scores, axis=0)
            std_importance = np.std(all_importance_scores, axis=0)
            
            # Create aggregated feature importance plot
            plt.figure(figsize=(12, 6))
            bars = plt.bar(feature_names, avg_importance, yerr=std_importance, capsize=5)
            plt.title('Average Feature Importance Across All Stocks', fontsize=14, pad=20)
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Average Importance Score', fontsize=12)
            plt.xticks(rotation=45)
        
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('aggregated_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print average feature importance
            print("\nAverage Feature Importance:")
            for name, score, std in zip(feature_names, avg_importance, std_importance):
                print(f"{name}: {score:.4f} (±{std:.4f})")
            
            # Calculate and visualize feature correlations
            if all_feature_values:
                # Convert feature values to DataFrame
                feature_df = pd.DataFrame(all_feature_values)
                
                # Calculate correlation matrix
                corr_matrix = feature_df.corr()
                
                # Create correlation heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, 
                           annot=True, 
                           cmap='coolwarm', 
                           center=0,
                           fmt='.2f',
                           square=True)
                plt.title('Feature Correlation Matrix', fontsize=14, pad=20)
                plt.tight_layout()
                plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Build portfolio using validation set data
        initial_cash = 1000000
        cash = initial_cash
        positions = {}
        daily_portfolio_values = {}
        
        # Get all unique dates from validation sets
        validation_dates = set()
        for data in all_val_data:
            dates_val, _, _, _ = data['validation_data']
            validation_dates.update(dates_val)
        validation_dates = sorted(validation_dates)
        
        validation_end_date = max(validation_dates)
        
        # Get all available dates up to latest date for portfolio tracking
        all_dates = pd.date_range(start=min(validation_dates), end=latest_date, freq='B').date
        all_dates = sorted(all_dates)
        
        # Process each day
        for current_date in tqdm(all_dates, desc="Building portfolio"):
            try:
                current_prices = {}
                current_predictions = {}
                
                # Get prices based on whether we're in validation period or not
                if current_date <= validation_end_date:
                    # During validation period, use validation data
                    for data in all_val_data:
                        symbol = data['symbol']
                        dates_val, prices_val, y_val, val_predictions = data['validation_data']
                        date_idx = np.where(dates_val == current_date)[0]
                        if len(date_idx) > 0:
                            idx = date_idx[0]
                            current_prices[symbol] = prices_val[idx]
                            current_predictions[symbol] = val_predictions[idx]
                else:
                    # After validation period, use cache data
                    current_date_dt = pd.to_datetime(current_date)
                    for symbol, df in stock_data_cache.items():
                        if current_date_dt in df.index:
                            current_prices[symbol] = df.loc[current_date_dt, 'close']
                        elif symbol in positions:
                            last_price = df['close'].iloc[-1] if not df.empty else None
                            if last_price is not None:
                                current_prices[symbol] = last_price
                
                # Skip if we don't have enough valid prices
                if not all(symbol in current_prices for symbol in positions):
                    continue
                
                # Check for positions to sell (held for one year)
                positions_to_sell = []
                for symbol, position_info in positions.items():
                    if (current_date - position_info['entry_date']).days >= 365:
                        positions_to_sell.append(symbol)
                
                # Sell positions
                for symbol in positions_to_sell:
                    position_info = positions[symbol]
                    quantity = position_info['quantity']
                    price = current_prices[symbol]
                    sale_value = quantity * price
                    cash += sale_value
                    
                    trade_history.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': quantity,
                        'price': price,
                        'value': sale_value,
                        'reason': 'HOLDING_PERIOD_END',
                        'prediction': current_predictions.get(symbol)
                    })
                    
                    del positions[symbol]
                
                # Only make new buy decisions during validation period
                if current_date <= validation_end_date:
                    selected_stocks = select_stocks(current_predictions, current_prices)
                    
                    # Buy new positions
                    for symbol in selected_stocks:
                        if symbol not in positions and symbol in current_prices:
                            price = current_prices[symbol]
                            quantity = int(cash * 0.1 / price)
                            if quantity > 0:
                                cost = quantity * price
                                if cost <= cash:
                                    cash -= cost
                                    positions[symbol] = {
                                        'quantity': quantity,
                                        'entry_date': current_date,
                                        'entry_price': price
                                    }
                                    
                                    trade_history.append({
                                        'date': current_date,
                                        'symbol': symbol,
                                        'action': 'BUY',
                                        'quantity': quantity,
                                        'price': price,
                                        'value': cost,
                                        'reason': 'NEW_POSITION',
                                        'prediction': current_predictions.get(symbol)
                                    })
                
                # Calculate and store portfolio value
                portfolio_value = calculate_portfolio_value(positions, cash, current_prices)
                daily_portfolio_values[current_date] = portfolio_value
                
            except Exception:
                continue
        
        # Convert portfolio values to DataFrame
        portfolio_df = pd.DataFrame(list(daily_portfolio_values.items()), columns=['date', 'value'])
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        initial_value = portfolio_df['value'].iloc[0]
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1
        
        # Calculate drawdown
        portfolio_df['peak'] = portfolio_df['value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['peak']) / portfolio_df['peak']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Print performance summary
        print("\nPortfolio Performance Summary:")
        print(f"Initial Value: ${initial_value:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"CAGR: {cagr:.2%}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        
        # Plot portfolio value over time
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df.index, portfolio_df['value'])
        plt.title('Portfolio Value Over Time', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('portfolio_value.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return

if __name__ == "__main__":
    main() 