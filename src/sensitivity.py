import torch
import numpy as np
import pandas as pd
from model import SimpleSCNN
from evaluate import run_test, apply_analog_noise, testloader # Reusing our eval logic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "./models/golden_snn_e4.pth"

# Define the search space
noise_levels = [0.02, 0.05, 0.10]
v_bounds = [1.1, 1.5, 2.0, 3.0, 5.0] # 1.1 is very tight (Vth=1.0), 5.0 is very loose

results = []

print(f"Starting Sensitivity Analysis on {len(noise_levels) * len(v_bounds)} combinations...")

for sigma in noise_levels:
    for v in v_bounds:
        # Initialize model with the specific bound we're testing
        model = SimpleSCNN(use_softsnn=True, beta=0.9).to(device)
        model.lif1.v_bound = v # Overriding the bound for each layer
        model.lif2.v_bound = v
        model.lif3.v_bound = v
        
        model.load_state_dict(torch.load(weights_path))
        model = apply_analog_noise(model, sigma=sigma)
        
        acc = run_test(model, testloader)
        results.append({"Sigma": sigma, "V_Bound": v, "Accuracy": acc})
        print(f"Sigma: {sigma} | V_Bound: {v} | Accuracy: {acc:.2f}%")

# Save to CSV for plotting
df = pd.DataFrame(results)
df.to_csv("./results/sensitivity_analysis.csv", index=False)