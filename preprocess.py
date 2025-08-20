import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from classification import GenderClassifierCNN, transform  # Ensure transform resizes to 224x224 and normalizes properly
import torch.nn as nn
import torch.optim as optim


class VideoDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Warning: Could not read image {img_path}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)  # fallback
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def load_data(frames_directory, labels=None, batch_size=32, test_split=0.2):
    image_paths = sorted(glob.glob(os.path.join(frames_directory, '**', '*.jpg'), recursive=True))
    if labels is None:
        labels = [int(os.path.basename(os.path.dirname(p))) for p in image_paths]

    dataset = VideoDataset(image_paths, labels)
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class VideoPreprocessor:
    def __init__(self, video_df):
        self.video_df = video_df
        self.frames_directory = 'extracted_frames'
        os.makedirs(self.frames_directory, exist_ok=True)
        self.frame_data = []

    def extract_frames(self):
        total_extracted = 0
        for index, row in self.video_df.iterrows():
            video_path = f"MultipleFiles/{row['id']}"
            if not os.path.exists(video_path):
                continue
            for filename in os.listdir(video_path):
                video_file = os.path.join(video_path, filename)
                total_extracted += self._extract_frames_from_video(video_file, row)
        self.save_frame_data_to_csv()
        print(f"📊 Total frames extracted: {total_extracted}")

    def _extract_frames_from_video(self, video_path, row):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0

        frame_count, extracted_count = 0, 0
        gender_dir = os.path.join(self.frames_directory, str(row['Gender']))
        os.makedirs(gender_dir, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 30 == 0:
                frame_filename = os.path.join(gender_dir, f"video_{row['id']}_frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                self.frame_data.append({'Video': video_path, 'Frame': frame_filename, 'Gender': row['Gender']})
                extracted_count += 1

            frame_count += 1

        cap.release()
        return extracted_count

    def save_frame_data_to_csv(self):
        frame_df = pd.DataFrame(self.frame_data)
        frame_df.to_csv('extracted_frames_data.csv', index=False)


class DataVisualizer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.frame_df = None

    def load_data(self):
        try:
            self.frame_df = pd.read_csv(self.csv_path)
            self.frame_df.columns = self.frame_df.columns.str.strip()
            return 'Gender' in self.frame_df.columns and 'Frame' in self.frame_df.columns
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False

    def visualize_data_distribution(self):
        if self.frame_df is not None:
            plt.figure(figsize=(8, 6))
            sns.countplot(x='Gender', data=self.frame_df, hue='Gender', palette='coolwarm', legend=False)
            plt.title("Gender Distribution")
            plt.xlabel("Gender")
            plt.ylabel("Count")
            plt.show()

    def display_sample_images(self):
        sample_df = self.frame_df.groupby('Gender').apply(lambda x: x.sample(min(5, len(x)))).reset_index(drop=True)
        fig, axes = plt.subplots(1, len(sample_df), figsize=(15, 5))
        for ax, (_, row) in zip(axes, sample_df.iterrows()):
            img = cv2.imread(row['Frame'])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(f"{row['Gender']}")
                ax.axis("off")
        plt.show()


def extract_video_features(frames_directory, sequence_length=10):
    sequences, labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_model = GenderClassifierCNN().to(device)
    cnn_model.eval()

    for gender_folder in os.listdir(frames_directory):
        folder_path = os.path.join(frames_directory, gender_folder)
        frame_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')])
        if len(frame_paths) < sequence_length:
            continue

        for i in range(0, len(frame_paths) - sequence_length + 1, sequence_length):
            frames = torch.stack([
                transform(cv2.cvtColor(cv2.imread(frame_paths[j]), cv2.COLOR_BGR2RGB))
                for j in range(i, i + sequence_length)
            ]).to(device)

            with torch.no_grad():
                features = cnn_model(frames).cpu().numpy()
            sequences.append(features)
            labels.append(int(gender_folder))

    return np.array(sequences), np.array(labels)


def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_acc_list, test_acc_list = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        test_accuracy = evaluate_model(model, test_loader, device)
        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

    return train_acc_list, test_acc_list


def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = (correct / total) * 100
    return accuracy
