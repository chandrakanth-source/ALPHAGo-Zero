import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual Block as specified in AlphaGo Zero (Silver et al. 2017).
    Contains 2 convolutional layers with Batch Normalization and ReLU activation,
    with a residual skip connection adding input x to the block output.
    """

    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class GoNetwork(nn.Module):
    """
    Dual-Head Residual Neural Network f_theta(s) = (p, v) for AlphaGo Zero.

    Architecture (Silver et al. 2017):
    - Input: 3 feature planes (board_size x board_size)
    - Initial Convolutional Block: 3x3 Conv -> BatchNorm -> ReLU
    - Residual Tower: 10 Residual Blocks
    - Policy Head: 1x1 Conv (2 filters) -> BatchNorm -> ReLU -> Linear -> Softmax
    - Value Head: 1x1 Conv (1 filter) -> BatchNorm -> ReLU -> Linear(64) -> ReLU -> Linear(1) -> Tanh
    """

    def __init__(self, board_size=12, num_res_blocks=10, num_channels=64):
        super().__init__()

        self.board_size = board_size
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels

        # Initial Convolutional Block
        self.initial_conv = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_channels)

        # Residual Tower (10 Residual Blocks)
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Backward compatibility aliases
        self.conv1 = self.initial_conv

        board_area = board_size * board_size

        # Policy Head (dual-head output 1)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_area, board_area + 1)
        )

        # Value Head (dual-head output 2)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_area, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.initial_bn(self.initial_conv(x)))

        for block in self.res_blocks:
            x = block(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        policy = F.softmax(policy, dim=1)

        return policy, value