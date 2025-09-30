import pandas as pd
from sqlalchemy import create_engine

# Import the machine learning tools we need from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

def main():
    """
    Main function to train and evaluate the baseline model.
    """
    # --- DATABASE CONNECTION (same as before) ---
    db_user = 'postgres'
    db_password = 'dheena46' # <-- IMPORTANT: REPLACE WITH YOUR PASSWORD
    db_host = 'localhost'
    db_port = '5432'
    db_name = 'movie_reviews'
    
    db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url)
    
    # 1. EXTRACT: Read the processed data from our database
    print("Reading processed data from database...")
    try:
        df = pd.read_sql('SELECT cleaned_text, sentiment FROM processed_reviews', engine)
        print(f"Successfully loaded {len(df)} processed reviews.")
    except Exception as e:
        print(f"Error reading from database: {e}")
        return

    # 2. SPLIT: Divide the data into training and testing sets
    # We'll use 80% of the data to train the model and 20% to test it.
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['sentiment'], 
        test_size=0.2, 
        random_state=42 # random_state ensures we get the same split every time
    )
    
    # 3. VECTORIZE: Convert text data into numerical vectors using TF-IDF
    # TF-IDF (Term Frequency-Inverse Document Frequency) is a standard way
    # to represent the importance of a word in a document.
    print("Vectorizing text data with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000) # Only consider the top 5000 words
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. TRAIN: Train the Logistic Regression model
    print("Training the Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    print("Model training complete.")

    # 5. EVALUATE: Make predictions and evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test_tfidf)
    
    print("\n--- Model Performance Report ---")
    print(classification_report(y_test, y_pred))
    print("--------------------------------\n")
    
    # 6. SAVE THE MODEL AND VECTORIZER
    # We save our trained model and the vectorizer so we can use them later
    # for making predictions without having to retrain.
    print("Saving the model and vectorizer to disk...")
    with open('sentiment_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
        
    with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
        
    print("Model and vectorizer saved successfully as 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl'.")


if __name__ == "__main__":
    main()