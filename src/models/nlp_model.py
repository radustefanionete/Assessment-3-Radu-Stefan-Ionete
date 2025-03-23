import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import classification_report

def train_text_model(texts, labels, epochs=3, batch_size=16):
    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    # Tokenizing the input text
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
    inputs = torch.tensor(encodings['input_ids'])
    attention_masks = torch.tensor(encodings['attention_mask'])
    labels = torch.tensor(labels)

    # Spliting the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(inputs, labels, test_size=0.1)

    # Creating DataLoader for training
    train_dataset = TensorDataset(X_train, attention_masks[X_train], y_train)
    val_dataset = TensorDataset(X_val, attention_masks[X_val], y_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setting up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # Training the model
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            b_input_ids, b_attention_mask, b_labels = batch
            optimizer.zero_grad()

            # Forward pass
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_dataloader)}")

    # Evaluating the model
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_dataloader:
            b_input_ids, b_attention_mask, b_labels = batch
            outputs = model(b_input_ids, attention_mask=b_attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)

            predictions.extend(predicted_labels.numpy())
            true_labels.extend(b_labels.numpy())

    # Printing classification report
    print(classification_report(true_labels, predictions, target_names=["negative", "neutral", "positive"]))
    return model
