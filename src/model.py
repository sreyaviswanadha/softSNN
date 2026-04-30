import torch
import torch.nn as nn
import snntorch as snn

class SoftSNNLeaky(snn.Leaky):
    """
    Custom Leaky Neuron with SoftSNN 'Bound-and-Protect' logic.
    Simulates analog clamping to mitigate voltage glitches.
    """
    def __init__(self, beta, threshold=1.0, v_bound=1.5, **kwargs):
        super().__init__(beta=beta, threshold=threshold, **kwargs)
        self.v_bound = v_bound 

    def forward(self, input_, mem=None):
        # Initialize membrane potential if it's the first time step
        if mem is None:
            mem = torch.zeros_like(input_)
            
        # 1. Update Membrane Potential (Analog RC integration)
        mem = (self.beta * mem) + input_
        
        # 2. SoftSNN 'Protect' Logic: Clamping
        # Mathematically models the physical voltage rails of an Analog IC
        mem = torch.clamp(mem, min=-self.v_bound, max=self.v_bound)
        
        # 3. Spike Generation using snnTorch's surrogate gradient
        spk = self.spike_grad(mem - self.threshold)
        
        # 4. Reset Mechanism (Standard 'subtract' reset)
        # In a real circuit, this is the charge discharge after firing
        mem = mem - (spk * self.threshold)
        
        return spk, mem

class SimpleSCNN(nn.Module):
    def __init__(self, beta=0.9, use_softsnn=False):
        super().__init__()
        # Use our custom SoftSNN neuron or the standard Leaky neuron
        neuron_layer = SoftSNNLeaky if use_softsnn else snn.Leaky
        
        self.conv1 = nn.Conv2d(2, 12, 5) 
        self.lif1 = neuron_layer(beta=beta)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.lif2 = neuron_layer(beta=beta)
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(32 * 5 * 5, 10)
        self.lif3 = neuron_layer(beta=beta)

    def forward(self, x):
        # Hidden states
        mem1, mem2, mem3 = None, None, None
        spk3_rec = []

        # Time-loop (Unrolling the SNN over time)
        for step in range(x.size(0)):
            cur1 = self.pool1(self.conv1(x[step]))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Flatten for the output linear layer
            cur3 = self.fc1(spk2.reshape(spk2.size(0), -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk3_rec.append(spk3)

        return torch.stack(spk3_rec, dim=0)