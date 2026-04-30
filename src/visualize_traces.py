import torch
import matplotlib.pyplot as plt
from model import SimpleSCNN
import tonic
import tonic.transforms as transforms

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "./models/golden_snn_e4.pth"
sigma = 0.10
v_bound = 3.0

# --- Data Probing ---
transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=1000),
])
testset = tonic.datasets.NMNIST(save_to='./data', train=False, transform=transform)
# Grab just one sample
data, target = testset[0] 
data = torch.tensor(data).float().unsqueeze(1).to(device) # Add batch dim: (Time, 1, C, H, W)

def get_traces(use_softsnn, inject_noise):
    model = SimpleSCNN(use_softsnn=use_softsnn).to(device)
    model.load_state_dict(torch.load(weights_path))
    
    if use_softsnn:
        model.lif1.v_bound = v_bound
        model.lif2.v_bound = v_bound
        model.lif3.v_bound = v_bound

    if inject_noise:
        with torch.no_grad():
            for param in model.parameters():
                if len(param.shape) > 1:
                    param.add_(torch.randn_like(param) * sigma)

    # Manual forward pass to record membrane potential of the 1st neuron in output layer
    model.eval()
    mem_trace = []
    mem1, mem2, mem3 = None, None, None
    
    with torch.no_grad():
        for step in range(data.size(0)):
            cur1 = model.pool1(model.conv1(data[step]))
            spk1, mem1 = model.lif1(cur1, mem1)
            cur2 = model.pool2(model.conv2(spk1))
            spk2, mem2 = model.lif2(cur2, mem2)
            cur3 = model.fc1(spk2.reshape(spk2.size(0), -1))
            spk3, mem3 = model.lif3(cur3, mem3)
            mem_trace.append(mem3[0, 0].item()) # Record 1st neuron
            
    return mem_trace

# --- Run the "Probes" ---
print("Probing internal voltages...")
trace_golden = get_traces(use_softsnn=False, inject_noise=False)
trace_unprotected = get_traces(use_softsnn=False, inject_noise=True)
trace_softsnn = get_traces(use_softsnn=True, inject_noise=True)

# --- Plotting ---
plt.figure(figsize=(12, 6))
time_steps = range(len(trace_golden))

plt.plot(time_steps, trace_golden, label='Golden (Ideal)', color='green', linewidth=2)
plt.plot(time_steps, trace_unprotected, label='Unprotected (Noisy)', color='red', alpha=0.6, linestyle='--')
plt.plot(time_steps, trace_softsnn, label='SoftSNN (Clamped @ 3.0V)', color='blue', linewidth=2)

plt.axhline(y=1.0, color='black', linestyle=':', label='Firing Threshold (Vth)')
plt.axhline(y=v_bound, color='blue', linestyle='-.', alpha=0.3, label='Voltage Rail (Vbound)')

plt.title(f"Neuron Membrane Potential ($V_{{mem}}$) Trace\nAnalog Noise $\sigma={sigma}$", fontsize=14)
plt.xlabel("Time Steps (ms)", fontsize=12)
plt.ylabel("Voltage (Normalized)", fontsize=12)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.savefig("./results/vmem_traces.png")
print("Trace visualization saved to ./results/vmem_traces.png")