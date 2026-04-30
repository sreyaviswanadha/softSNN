import torch
import torch.nn as nn
import snntorch.functional as SF
import tonic
import tonic.transforms as transforms
from model import SimpleSCNN
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "./models/golden_snn_e4.pth"
os.makedirs('./results', exist_ok=True)

transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=1000),
])
testset = tonic.datasets.NMNIST(save_to='./data', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, collate_fn=tonic.collation.PadTensors())

# --- 2. Helper Functions ---
def apply_layer_noise(model, target_layer_name, sigma=0.1):
    """Applies noise ONLY to the specified layer's weights."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if target_layer_name in name and len(param.shape) > 1:
                noise = torch.randn_like(param) * sigma
                param.add_(noise)
    return model

def evaluate_and_collect(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in loader:
            data = data.transpose(0, 1).to(device)
            targets = targets.to(device)
            spk_rec = model(data)
            
            # Get predictions (class with highest spike count)
            _, idx = spk_rec.sum(dim=0).max(1)
            all_preds.extend(idx.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            correct += (idx == targets).sum().item()
            total += targets.size(0)
    return (correct / total) * 100, all_targets, all_preds

# --- 3. Run Layer-wise Sensitivity Analysis ---
layers_to_test = ['conv1', 'conv2', 'fc1']
sigma_test = 0.1
sensitivity_results = []

print(f"Starting Layer-wise Sensitivity (Sigma={sigma_test})...")
for layer in layers_to_test:
    # Test Standard
    m_std = SimpleSCNN(use_softsnn=False).to(device)
    m_std.load_state_dict(torch.load(weights_path))
    m_std = apply_layer_noise(m_std, layer, sigma=sigma_test)
    acc_std, _, _ = evaluate_and_collect(m_std, testloader)
    
    # Test SoftSNN (Optimized Bound)
    m_soft = SimpleSCNN(use_softsnn=True).to(device)
    m_soft.load_state_dict(torch.load(weights_path))
    m_soft.lif1.v_bound, m_soft.lif2.v_bound, m_soft.lif3.v_bound = 3.0, 3.0, 3.0
    m_soft = apply_layer_noise(m_soft, layer, sigma=sigma_test)
    acc_soft, _, _ = evaluate_and_collect(m_soft, testloader)
    
    sensitivity_results.append({'Layer': layer, 'Standard': acc_std, 'SoftSNN': acc_soft})
    print(f"Fault in {layer}: Standard={acc_std:.2f}%, SoftSNN={acc_soft:.2f}%")

# Save Sensitivity Table
df_sens = pd.DataFrame(sensitivity_results)
df_sens.to_csv('./results/layer_sensitivity.csv', index=False)

# --- 4. Generate Confusion Matrices (At Full Sigma=0.1) ---
print("\nGenerating Confusion Matrices at Sigma=0.1...")
full_sigma = 0.1

def plot_cm(targets, preds, title, filename):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'./results/{filename}')
    plt.close()

# Standard Global Noise
m_std = SimpleSCNN(use_softsnn=False).to(device)
m_std.load_state_dict(torch.load(weights_path))
with torch.no_grad():
    for p in m_std.parameters():
        if len(p.shape) > 1: p.add_(torch.randn_like(p) * full_sigma)
_, t_std, p_std = evaluate_and_collect(m_std, testloader)
plot_cm(t_std, p_std, "Confusion Matrix: Standard SNN (Sigma=0.1)", "cm_standard.png")

# SoftSNN Global Noise
m_soft = SimpleSCNN(use_softsnn=True).to(device)
m_soft.load_state_dict(torch.load(weights_path))
m_soft.lif1.v_bound, m_soft.lif2.v_bound, m_soft.lif3.v_bound = 3.0, 3.0, 3.0
with torch.no_grad():
    for p in m_soft.parameters():
        if len(p.shape) > 1: p.add_(torch.randn_like(p) * full_sigma)
_, t_soft, p_soft = evaluate_and_collect(m_soft, testloader)
plot_cm(t_soft, p_soft, "Confusion Matrix: SoftSNN (Sigma=0.1, Vbound=3.0)", "cm_softsnn.png")

print("Analysis Complete. Check the ./results folder for CSV and PNGs.")