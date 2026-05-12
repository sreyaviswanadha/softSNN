# SoftSNN-Inspired Bound-and-Protect for Robust SNN Inference

This project implements and evaluates a SoftSNN-inspired robustness mechanism for Spiking Neural Networks (SNNs). The goal is to study how membrane-potential bounding and Bound-and-Protect-style fault mitigation improve SNN reliability under hardware-inspired noise.

## Project Structure

```text
models/
  golden_snn_e0.pth - golden model checkpoint after epoch 0
  golden_snn_e1.pth - golden model checkpoint after epoch 1
  golden_snn_e2.pth - golden model checkpoint after epoch 2
  golden_snn_e3.pth - golden model checkpoint after epoch 3
  golden_snn_e4.pth - final trained golden model checkpoint

results/
  cm_softsnn.png              - confusion matrix for SoftSNN-protected model
  cm_standard.png             - confusion matrix for standard noisy SNN
  layer_sensitivity.csv       - layer-wise robustness comparison
  sensitivity_analysis.csv    - accuracy sweep over sigma and V_bound
  vmem_traces.png             - membrane potential trace comparison

src/
  dataset.py                        - N-MNIST dataset loading and preprocessing
  model.py                          - standard SNN and SoftSNN-style LIF model
  train.py                          - trains the clean golden SNN model
  evaluate.py                       - evaluates standard and protected models under noise
  sensitivity.py                    - runs V_bound and noise-level sensitivity analysis
  visualize_traces.py               - generates membrane potential trace plots
  analysis.py                       - additional result analysis utilities
  SoftSNN_Lite_Bound_and_Protect... - MNIST BnP experiment notebook
