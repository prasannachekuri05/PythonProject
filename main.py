import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from classification import (GenderClassifierCNN, RNNModel,
                            DeepANNModel, AlexNetModel, ANNModel, LSTMModel, VGGModel, iAutoencoder)
from preprocess import VideoDataset, VideoPreprocessor


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()
    for epoch in range(epochs):

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # Evaluation on test set
    accuracy = evaluate_model(model, test_loader, device)
    print(f"Final Test Accuracy: {accuracy:.2f}%")

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("Classification Report:\n", classification_report(all_labels, all_preds, zero_division=1))


def main():
    csv_path = 'extracted_frames_data.csv'

    # print("🔄 Loading video dataset...")
    video_df = pd.read_csv('MultipleFiles/extracted_frames_data.csv', delimiter=';')
    print(f"📄 Columns in video_df: {video_df.columns.tolist()}")

    preprocessor = VideoPreprocessor(video_df)
    preprocessor.extract_frames()
    preprocessor.save_frame_data_to_csv()

    # print("📂 Loading extracted frame data...")
    frame_df = pd.read_csv(csv_path)
    images, labels = frame_df['Frame'], frame_df['Gender']

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    dataset = VideoDataset(images, encoded_labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    models = {
        "CNN": GenderClassifierCNN(),
        "RNN": RNNModel(),
        "Deep ANN": DeepANNModel(),
        "ANN": ANNModel(),
        "LSTM": LSTMModel(),
        "VGG": VGGModel(),
        "AlexNet": AlexNetModel(),
        # "AutoEncoder": iAutoencoder()
    }

    for name, model in models.items():
        print(f"🛠️ Training {name} model...")
        train_model(model, train_loader, test_loader, epochs=3, lr=0.001)
        print(f"✅ {name} Training Completed!")


if __name__ == "__main__":
    main()
