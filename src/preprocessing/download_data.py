import os
import zipfile

# Defining the local dataset path
local_zip_path = r"C:\Users\stefa\OneDrive\Documents\Stefan\OPIT - MSc Data Science & AI\Workings\Applications in Data Science & AI\twitter_sentiment_data.zip"
extract_path = "data/raw"

def extract_local_data():
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    try:
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Dataset extracted successfully to {extract_path}")
    except Exception as e:
        print(f"Error extracting dataset: {e}")

if __name__ == "__main__":
    extract_local_data()
