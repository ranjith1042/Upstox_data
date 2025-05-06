import sqlite3

def clean_financial_data():
    """Clean and reformat financial data in the database"""
    try:
        conn = sqlite3.connect('financial_data.db')
        cursor = conn.cursor()
        
        # Get all quarterly income statement tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'IS_%_Q_C'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            print(f"\nCleaning table: {table_name}")
            
            # Get the data
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            if not rows:
                continue
                
            # Get column names
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Create a new table with cleaned data
            new_table_name = f"{table_name}_CLEAN"
            cursor.execute(f"DROP TABLE IF EXISTS {new_table_name}")
            
            # Create the new table with proper columns
            create_table_sql = f"""
            CREATE TABLE {new_table_name} (
                Metric TEXT,
                {', '.join([f'"{col}" REAL' for col in columns[1:]])}
            )
            """
            cursor.execute(create_table_sql)
            
            # Process each row
            for row in rows:
                metric = row[0]
                values = row[1:]
                
                # Clean and convert values
                cleaned_values = []
                for val in values:
                    if isinstance(val, str):
                        # Remove any non-numeric characters except decimal point and minus sign
                        cleaned = ''.join(c for c in val if c.isdigit() or c in '.-')
                        try:
                            # Convert to float, handling cases like '1,234.56'
                            cleaned = float(cleaned.replace(',', ''))
                        except ValueError:
                            cleaned = None
                    else:
                        cleaned = val
                    cleaned_values.append(cleaned)
                
                # Insert cleaned data
                placeholders = ', '.join(['?'] * (len(columns)))
                insert_sql = f"INSERT INTO {new_table_name} VALUES ({placeholders})"
                cursor.execute(insert_sql, [metric] + cleaned_values)
            
            # Replace the old table with the cleaned one
            cursor.execute(f"DROP TABLE {table_name}")
            cursor.execute(f"ALTER TABLE {new_table_name} RENAME TO {table_name}")
            
            print(f"Successfully cleaned {table_name}")
        
        conn.commit()
        conn.close()
        print("\nAll tables cleaned successfully!")
        
    except Exception as e:
        print(f"Error cleaning financial data: {str(e)}")
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    clean_financial_data() 