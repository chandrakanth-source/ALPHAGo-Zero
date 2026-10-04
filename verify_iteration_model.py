import torch

from network.model_loader import load_model


model_path = "models/model_iteration_1.pt"

print("=" * 60)
print("VERIFYING ITERATION 1 MODEL")
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