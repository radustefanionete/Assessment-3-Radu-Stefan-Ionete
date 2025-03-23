import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# Load text model & vectorizer
text_model = joblib.load("train/models/text_model.pkl")
vectorizer = joblib.load("train/models/vectorizer.pkl")

# Load image model
image_model = models.resnet50(pretrained=False)
num_ftrs = image_model.fc.in_features
image_model.fc = nn.Linear(num_ftrs, 3)
image_model.load_state_dict(torch.load("train/models/image_model.pth"))
image_model.eval()

# Dummy function to extract image features
def extract_image_features(image_path):
    return np.random.rand(2048)  # Replace with actual feature extraction

# Load dataset
df = pd.read_csv("processed_data/combined_dataset.csv")

# Process text and image features
text_features = vectorizer.transform(df["cleaned_text"]).toarray()
image_features = np.array([extract_image_features(path) for path in df["image_path"]])
X = np.hstack([text_features, image_features])
y = df["sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define final classifier
class FusionModel(nn.Module):
    def __init__(self, input_size):
        super(FusionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 sentiment classes
        )
    
    def forward(self, x):
        return self.fc(x)

# Initializing model
fusion_model = FusionModel(input_size=X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.0001)

# Converting to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Training loop
epochs = 5
for epoch in range(epochs):
    fusion_model.train()
    optimizer.zero_grad()
    outputs = fusion_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Saving final model
os.makedirs("train/models", exist_ok=True)
torch.save(fusion_model.state_dict(), "train/models/fusion_model.pth")

print("Final multi-modal sentiment model saved successfully.")
