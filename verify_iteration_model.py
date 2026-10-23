import sys
import torch

from network.model_loader import load_model


model_path = sys.argv[1] if len(sys.argv) > 1 else "models/model_iteration_10.pt"

print("=" * 60)
print(f"VERIFYING MODEL: {model_path}")
print("=" * 60)

model = load_model(
    model_path,
    board_size=12
)

model.eval()

print("Model loaded successfully.")

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    "Total parameters:",
    parameter_count
)

print("=" * 60)