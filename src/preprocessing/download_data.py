import zipfile
import os

# Path to the dataset zip file in Google Colab
local_zip_path = '/content/twitter_sentiment_data.zip'
extract_path = '/content/data/raw'

def extract_local_data():
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    try:
        # Extracting the dataset
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Dataset extracted successfully to {extract_path}")
    except Exception as e:
        print(f"Error extracting dataset: {e}")

# Running the extraction function
if __name__ == "__main__":
    extract_local_data()
