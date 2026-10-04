import torch


path = "data/self_play_iteration_1.pt"

data = torch.load(
    path,
    weights_only=False
)

print("=" * 60)
print("SELF-PLAY DATASET INSPECTION")
print("=" * 60)

print("Type:", type(data))
print("Number of examples:", len(data))

if len(data) > 0:

    print()
    print("First example:")
    print(data[0])

print("=" * 60)