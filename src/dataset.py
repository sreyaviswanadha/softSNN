import torch
import snntorch as snn
import tonic
import matplotlib.pyplot as plt

# 1. Dataset Setup (N-MNIST)
# Tonic handles the spike-to-tensor conversion
sensor_size = tonic.datasets.NMNIST.sensor_size
transform = tonic.transforms.Compose([
    tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000), # 1ms bins
    tonic.transforms.Denoise(filter_time=10000),
])

trainset = tonic.datasets.NMNIST(save_to='./data', train=True, transform=transform)
# Note: For the 1-week sprint, we'll start with a subset for speed
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=tonic.collation.PadTensors())

# 2. Define the Neural Dynamics (LaTeX Equation for your report)
# V_mem[t+1] = \beta V_mem[t] + (1-\beta) I_{in}[t+1] - V_{th} S[t]
beta = 0.9  # Neuron decay rate (RC time constant proxy)
v_th = 1.0  # Firing threshold

# Initialize a single Leaky Integrate-and-Fire neuron
lif = snn.Leaky(beta=beta, threshold=v_th)

print("Setup Complete. Ready for training.")