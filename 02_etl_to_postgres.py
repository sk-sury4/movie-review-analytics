import pandas as pd
from sqlalchemy import create_engine
import time

def main():
    """
    Main function to run the ETL process.
    """
    # --- DATABASE CONNECTION DETAILS ---
    # Replace with your PostgreSQL password
    # Default username and database name are often 'postgres' if you didn't change them
    db_user = 'postgres'
    db_password = 'dheena46' # <-- IMPORTANT: REPLACE WITH YOUR PASSWORD
    db_host = 'localhost'
    db_port = '5432' # The default port you chose during installation
    db_name = 'movie_reviews'

    # Create the database connection URL
    # This format is standard for SQLAlchemy
    db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    print("Connecting to the database...")
    try:
        engine = create_engine(db_url)
        # Test the connection
        connection = engine.connect()
        print("Database connection successful!")
        connection.close()
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return # Exit the function if connection fails

    # --- EXTRACT ---
    print("Step 1: Extracting data from CSV...")
    try:
        df = pd.read_csv("IMDB Dataset.csv")
        print(f"Extracted {len(df)} rows from CSV.")
    except FileNotFoundError:
        print("Error: 'IMDB Dataset.csv' not found.")
        return

    # --- TRANSFORM ---
    # A simple transformation: rename columns to match our database table
    print("Step 2: Transforming data...")
    df.rename(columns={'review': 'review_text'}, inplace=True)
    print("Columns renamed.")

    # --- LOAD ---
    print("Step 3: Loading data into PostgreSQL...")
    start_time = time.time()
    try:
        # Use pandas `to_sql` function to load the dataframe into the 'reviews' table
        # `if_exists='replace'` will drop the table if it already exists and create a new one.
        # This is useful for development so you can re-run the script easily.
        df.to_sql(
            'reviews',
            engine,
            if_exists='replace',
            index=False # Don't write the pandas DataFrame index as a column
        )
        end_time = time.time()
        print(f"Data loaded successfully into 'reviews' table in {end_time - start_time:.2f} seconds.")

    except Exception as e:
        print(f"Error loading data into the database: {e}")


if __name__ == "__main__":
    main()