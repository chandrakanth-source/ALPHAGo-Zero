import torch
import torch.nn as nn
import torch.nn.functional as F


class GoNetwork(nn.Module):

    def __init__(self, board_size=12):
        super().__init__()

        self.board_size = board_size

        self.initial_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1 = self.initial_conv
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        board_area = board_size * board_size

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_area, board_area + 1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_area, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):

        x = F.relu(self.initial_conv(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        policy = self.policy_head(x)

        value = self.value_head(x)

        policy = F.softmax(policy, dim=1)

        return policy, value