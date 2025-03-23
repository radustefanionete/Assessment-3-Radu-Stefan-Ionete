import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf

# Load test data
X_text_test = joblib.load("train/X_text_test.pkl")
X_image_test = joblib.load("train/X_image_test.pkl")
y_test = joblib.load("train/y_test.pkl")

# Loading trained models
text_model = joblib.load("train/text_model.pkl")
image_model = tf.keras.models.load_model("train/image_model.h5")
final_model = joblib.load("train/final_model.pkl")

# Predicting text-based sentiment
text_preds = text_model.predict(X_text_test)

# Predicting image-based sentiment
image_preds = image_model.predict(X_image_test)
image_preds = np.argmax(image_preds, axis=1)  # Convert softmax outputs to labels

# Combining predictions
combined_preds = final_model.predict(np.hstack((text_preds.reshape(-1, 1), image_preds.reshape(-1, 1))))

# Evaluating model performance
accuracy = accuracy_score(y_test, combined_preds)
print(f"Model Accuracy: {accuracy:.4f}\n")

# Classification report
print("Classification Report:\n", classification_report(y_test, combined_preds))

# Confusion matrix
cm = confusion_matrix(y_test, combined_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
