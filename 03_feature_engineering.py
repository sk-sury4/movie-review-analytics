import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time

# --- One-time setup for NLTK ---
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK resources (stopwords and punkt)...")
    nltk.download('stopwords')
    nltk.download('punkt')
    print("NLTK resources downloaded.")

def clean_text(text):
    """
    Cleans the input text.
    """
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return " ".join(filtered_text)

def main():
    """
    Main function to run the feature engineering pipeline.
    """
    db_user = 'postgres'
    db_password = 'dheena46' # <-- IMPORTANT: REPLACE WITH YOUR PASSWORD
    db_host = 'localhost'
    db_port = '5432'
    db_name = 'movie_reviews'
    
    db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url)
    
    print("Connecting to database and reading raw reviews...")
    try:
        # --- CHANGE 1: We are no longer asking for an 'id' column that doesn't exist. ---
        df = pd.read_sql('SELECT review_text, sentiment FROM reviews', engine)
        print(f"Successfully extracted {len(df)} rows.")
    except Exception as e:
        print(f"Error reading from database: {e}")
        return

    print("Starting text cleaning process (this may take a few minutes)...")
    start_time = time.time()
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    end_time = time.time()
    print(f"Text cleaning completed in {end_time - start_time:.2f} seconds.")
    
    print("\n--- Sample of cleaned data ---")
    # --- CHANGE 2: We only show the columns that exist now ---
    print(df[['cleaned_text', 'sentiment']].head())
    print("----------------------------\n")

    print("Loading processed data into 'processed_reviews' table...")
    try:
        # --- CHANGE 3: We need to create the 'id' column ourselves before loading. ---
        # The database needs a primary key. We can use the pandas index for this.
        processed_df = df[['cleaned_text', 'sentiment']].copy()
        processed_df.reset_index(inplace=True) # This turns the index into a column
        processed_df.rename(columns={'index': 'id'}, inplace=True) # Rename it to 'id'
        
        processed_df.to_sql(
            'processed_reviews',
            engine,
            if_exists='replace',
            index=False
        )
        print("Processed data loaded successfully!")
    except Exception as e:
        print(f"Error loading processed data: {e}")

if __name__ == "__main__":
    main()
