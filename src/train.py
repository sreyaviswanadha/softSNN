import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
import tonic
import tonic.transforms as transforms
import os
from model import SimpleSCNN

# --- 1. Hyperparameters & Setup ---
batch_size = 64
learning_rate = 1e-3
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories if they don't exist
os.makedirs('./models', exist_ok=True)
os.makedirs('./data', exist_ok=True)

# --- 2. Dataset & Transform ---
sensor_size = tonic.datasets.NMNIST.sensor_size
transform = transforms.Compose([
    transforms.Denoise(filter_time=10000), 
    transforms.ToFrame(sensor_size=sensor_size, time_window=1000), 
])

# Initialize Dataset
dataset = tonic.datasets.NMNIST(save_to='./data', train=True, transform=transform)

# Use Disk Cache to speed up training after the first epoch
cached_set = tonic.DiskCachedDataset(dataset, cache_path='./cache/nmnist/train')

trainloader = torch.utils.data.DataLoader(
    cached_set, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=tonic.collation.PadTensors()
)

# --- 3. Model, Optimizer, Loss ---
# use_softsnn=False because this is our "Golden" baseline training
model = SimpleSCNN(use_softsnn=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# MSE Count Loss: Encourages correct class to fire more spikes
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

# --- 4. Training Loop ---
print(f"Starting Training on {device}...")

for epoch in range(num_epochs):
    avg_loss = 0
    for i, (data, targets) in enumerate(trainloader):
        # data shape: (batch, time, channel, h, w)
        # SNNTorch expects: (time, batch, channel, h, w)
        data = data.transpose(0, 1).to(device)
        targets = targets.to(device)

        model.train()
        spk_rec = model(data) 
        
        loss_val = loss_fn(spk_rec, targets)
        
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        avg_loss += loss_val.item()
        if i % 10 == 0:
            print(f"Epoch {epoch}, Iter {i}, Loss: {loss_val.item():.4f}")

    # Save the 'Golden' weights after each epoch
    save_path = f"./models/golden_snn_e{epoch}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Epoch {epoch} complete. Saved to {save_path}")

print("Training Complete. All Golden weights saved.")