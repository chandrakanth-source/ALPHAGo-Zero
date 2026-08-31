import torch
from torch.utils.data import Dataset


class SelfPlayDataset(Dataset):

    def __init__(self, examples):

        self.examples = examples

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, index):

        state, policy, value = self.examples[index]

        state = torch.tensor(
            state,
            dtype=torch.float32
        )

        policy = torch.tensor(
            policy,
            dtype=torch.float32
        )

        value = torch.tensor(
            value,
            dtype=torch.float32
        )

        return state, policy, value