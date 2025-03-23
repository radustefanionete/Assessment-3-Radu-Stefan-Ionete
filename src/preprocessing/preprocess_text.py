import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Downloading NLTK data (tokenizer, stopwords)
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text: Tokenization, removing stopwords, lowercasing
def preprocess_text(text):
    # Remove non-alphabetic characters (e.g., punctuation, numbers)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Converting to lowercase
    text = text.lower()
    
    # Tokenizing the text
    tokens = nltk.word_tokenize(text)
    
    # Removing stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stopwords]
    
    # Joining tokens back into a string
    return ' '.join(tokens)

# Function to load dataset and preprocess (for Excel file)
def load_and_preprocess_data(excel_file):
    # Load dataset from Excel
    df = pd.read_excel(excel_file)
    
    # Preprocessing the text data
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Splitting dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['sentiment'], test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Function to extract features using TF-IDF
def extract_features(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=10000)  # Limit to 10,000 features
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf

# Main function to run the preprocessing steps
if __name__ == '__main__':
    # Path to the dataset Excel file in Google Drive
    excel_file = '/content/drive/MyDrive/datasets/twitter_sentiment_data/LabeledData.xlsx'  # Adjust this path
    
    # Loading and preprocessing data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(excel_file)
    
    # Extracting features using TF-IDF
    X_train_tfidf, X_test_tfidf = extract_features(X_train, X_test)
    
    # Output the shape of the resulting feature matrices
    print(f"Training data shape: {X_train_tfidf.shape}")
    print(f"Test data shape: {X_test_tfidf.shape}")
    
    # Saving the processed data and features for later use
    import joblib
    joblib.dump(vectorizer, '/content/drive/MyDrive/datasets/vectorizer.pkl')
    joblib.dump(X_train_tfidf, '/content/drive/MyDrive/datasets/X_train_tfidf.pkl')
    joblib.dump(X_test_tfidf, '/content/drive/MyDrive/datasets/X_test_tfidf.pkl')
