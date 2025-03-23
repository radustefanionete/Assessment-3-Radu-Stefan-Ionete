import gdown
import zipfile
import os

# Google Drive File ID
file_id = '1ZAi_M9VnMqGGiWprFcXVaGecawFlolcU'  
url = f'https://drive.google.com/file/d/1ZAi_M9VnMqGGiWprFcXVaGecawFlolcU/view?usp=drive_link'

# Output directory for the downloaded zip file
output = 'data/twitter_sentiment_data.zip'

# Downloading the dataset from Google Drive
gdown.download(url, output, quiet=False)

# Extracting the dataset
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('data/')

print("Dataset downloaded and extracted successfully!")
