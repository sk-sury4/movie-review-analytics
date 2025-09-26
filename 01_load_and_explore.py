import pandas as pd

# This function will be the start of your entire project.
def load_initial_data():
    """
    Loads the raw IMDb dataset from the CSV file.
    """
    print("Loading raw IMDb dataset...")
    try:
        # The file is in the same folder, so we can just use its name.
        df = pd.read_csv("IMDB Dataset.csv")
        print("Dataset loaded successfully!")
        print("----------------------------------------")
        print("Dataset Information:")
        df.info()
        print("----------------------------------------")
        print("First 5 rows of the dataset:")
        print(df.head())
        print("----------------------------------------")
        print(f"Total number of reviews: {len(df)}")
        return df
    except FileNotFoundError:
        print("Error: 'IMDB Dataset.csv' not found. Make sure it's in the same folder as this script.")
        return None

# This part makes the script runnable
if __name__ == "__main__":
    raw_df = load_initial_data()