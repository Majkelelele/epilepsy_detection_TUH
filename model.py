import os
import glob
import numpy as np
import pyedflib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# Directories
eps_dir = "00_epilepsy"
noEps_dir = "01_no_epilepsy"

# Parameters
FIXED_CHANNELS = 99  
FIXED_LENGTH = 1000  

def read_edf(file_name):
    f = pyedflib.EdfReader(file_name)
    signals = np.zeros((FIXED_CHANNELS, FIXED_LENGTH))  # (channels, time)
    
    for i, label in enumerate(f.getSignalLabels()):
        if i >= FIXED_CHANNELS:
            break  # Ignore extra channels
        
        signal = f.readSignal(i)
        signal = resample(signal, FIXED_LENGTH)  
        signals[i, :] = signal 
    
    f.close()
    return signals

class EEGDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels[idx]
        
        data = read_edf(file)
        data = (data - np.mean(data)) / np.std(data)  
        data = torch.tensor(data, dtype=torch.float32)  
        return data.unsqueeze(0), torch.tensor(label, dtype=torch.long) 

def load_data():
    files, labels = [], []

    for file in glob.glob(f"{eps_dir}/**/*.edf", recursive=True):
        files.append(file)
        labels.append(1)  # Epilepsy

    for file in glob.glob(f"{noEps_dir}/**/*.edf", recursive=True):
        files.append(file)
        labels.append(0)  # No epilepsy

    return train_test_split(files, labels, test_size=0.2, random_state=42)

train_files, test_files, train_labels, test_labels = load_data()

train_dataset = EEGDataset(train_files, train_labels)
test_dataset = EEGDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


class EEGCNN(nn.Module):
    def __init__(self):
        super(EEGCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

        with torch.no_grad():
            sample_input = torch.randn(1, 1, FIXED_CHANNELS, FIXED_LENGTH)  
            sample_output = self.pool(torch.relu(self.conv2(self.pool(torch.relu(self.conv1(sample_input))))))
            self.flattened_size = sample_output.view(1, -1).size(1) 

        self.fc1 = nn.Linear(self.flattened_size, 128) 
        self.fc2 = nn.Linear(128, 1)  

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.flatten(x) 
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x.squeeze(1)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EEGCNN().to(device)
criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as tepoch:
        for inputs, labels in tepoch:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            tepoch.set_postfix(loss=running_loss/len(tepoch))
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")


model.eval()
correct, total = 0, 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)
        outputs = model(inputs)
        predictions = (outputs > 0.5).float()
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")


