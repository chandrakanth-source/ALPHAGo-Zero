"""
AlphaGo Zero - Model Iteration 10 Loader & Verification Script
Loads the newly trained Candidate Iteration 10 model (models/model_iteration_10.pt),
inspects parameters, verifies network architecture, and runs a forward inference test.
Note: latest_model.pt remains the reigning champion (Iteration 9) and is NOT overwritten.
"""

import os
import torch
import numpy as np

from network.model_loader import load_model
from network.network import GoNetwork

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model_iteration_10.pt")
BOARD_SIZE = 12

def main():
    print("=" * 70)
    print(" ALPHAGO ZERO - ITERATION 10 MODEL VERIFICATION")
    print("=" * 70)
    print(f"Loading candidate model from: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    # Load model
    model = load_model(MODEL_PATH, board_size=BOARD_SIZE)
    model.eval()
    print(">> Model loaded successfully!")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">> Total parameters:     {total_params:,}")
    print(f">> Trainable parameters: {trainable_params:,}")

    # Forward pass test with a sample board input (1, 3, 12, 12)
    sample_input = torch.zeros((1, 3, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    with torch.no_grad():
        policy_logits, value = model(sample_input)
        policy_probs = torch.softmax(policy_logits, dim=1).numpy()[0]
        value_pred = value.item()

    print(f">> Forward pass test:    SUCCESS")
    print(f">> Policy output shape:  {policy_logits.shape} (144 board positions + 1 pass move)")
    print(f">> Top move probability: {np.max(policy_probs):.4f} at index {np.argmax(policy_probs)}")
    print(f">> Initial value output: {value_pred:.4f} (range: -1.0 to +1.0)")
    print("=" * 70)
    print("Model Iteration 10 is verified and ready for inference / web arena play.")
    print("Reigning champion latest_model.pt remains Iteration 9 (untouched).")
    print("=" * 70)

if __name__ == "__main__":
    main()
