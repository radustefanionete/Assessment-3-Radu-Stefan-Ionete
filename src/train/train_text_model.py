import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Loading preprocessed text data
df = pd.read_csv("processed_data/processed_text.csv")

# Vectorizing text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_text"])
y = df["sentiment"]

# Spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluating model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Saving model and vectorizer
os.makedirs("train/models", exist_ok=True)
joblib.dump(model, "train/models/text_model.pkl")
joblib.dump(vectorizer, "train/models/vectorizer.pkl")

print("Text sentiment model saved successfully.")
