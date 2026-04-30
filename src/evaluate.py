import torch
import snntorch.functional as SF
import tonic
import tonic.transforms as transforms
from model import SimpleSCNN
import os

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sensor_size = tonic.datasets.NMNIST.sensor_size
transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
])

# Ensure data path is correct
data_path = './data'
testset = tonic.datasets.NMNIST(save_to=data_path, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, collate_fn=tonic.collation.PadTensors())

def apply_analog_noise(model, sigma=0.01):
    """Adds Gaussian noise to weights to simulate device mismatch/variation."""
    with torch.no_grad():
        for param in model.parameters():
            if len(param.shape) > 1:
                noise = torch.randn_like(param) * sigma
                param.add_(noise)
    return model

def run_test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in loader:
            # SNN expects (Time, Batch, Channel, H, W)
            data = data.transpose(0, 1).to(device)
            targets = targets.to(device)
            spk_rec = model(data)
            # accuracy_rate calculates spikes per class over time
            correct += SF.accuracy_rate(spk_rec, targets) * targets.size(0)
            total += targets.size(0)
    return (correct / total) * 100

# --- The Experiment ---
weights_path = "./models/golden_snn_e4.pth"

if not os.path.exists(weights_path):
    print(f"Error: Could not find {weights_path}. Please run training first.")
    exit()

print(f"{'Fault Sigma':<12} | {'Standard SNN (%)':<18} | {'SoftSNN (%)'}")
print("-" * 50)

for sigma in [0.0, 0.02, 0.05, 0.1, 0.15]:
    # Standard SNN (Vulnerable)
    model_std = SimpleSCNN(use_softsnn=False).to(device)
    model_std.load_state_dict(torch.load(weights_path))
    model_std = apply_analog_noise(model_std, sigma)
    acc_std = run_test(model_std, testloader)
    
    # SoftSNN (Protected by Clamping)
    model_soft = SimpleSCNN(use_softsnn=True).to(device)
    model_soft.load_state_dict(torch.load(weights_path))
    model_soft = apply_analog_noise(model_soft, sigma)
    acc_soft = run_test(model_soft, testloader)
    
    print(f"{sigma:<12.2f} | {acc_std:<18.2f} | {acc_soft:.2f}")