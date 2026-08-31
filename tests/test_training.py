import numpy as np

from training.dataset import SelfPlayDataset


def test_dataset():

    examples = [

        (
            np.zeros(
                (3, 19, 19),
                dtype=np.float32
            ),

            np.ones(
                362,
                dtype=np.float32
            ) / 362,

            1.0
        )

    ]

    dataset = SelfPlayDataset(
        examples
    )

    assert len(dataset) == 1

    state, policy, value = dataset[0]

    assert state.shape == (3, 19, 19)

    assert policy.shape == (362,)

    assert value.item() == 1.0
import torch

from network.network import GoNetwork
from training.trainer import Trainer


def test_training_loss():

    model = GoNetwork(
        board_size=19
    )

    trainer = Trainer(
        model
    )

    predicted_policy = torch.randn(
        2,
        362
    )

    predicted_value = torch.randn(
        2,
        1
    )

    target_policy = torch.softmax(
        torch.randn(
            2,
            362
        ),
        dim=1
    )

    target_value = torch.tensor(
        [1.0, -1.0]
    )

    (
        loss,
        policy_loss,
        value_loss
    ) = trainer.calculate_loss(
        predicted_policy,
        predicted_value,
        target_policy,
        target_value
    )

    assert loss.item() >= 0

    assert policy_loss.item() >= 0

    assert value_loss.item() >= 0
def test_training_updates_weights():

    import torch
    from torch.utils.data import DataLoader

    from network.network import GoNetwork
    from training.dataset import SelfPlayDataset
    from training.trainer import Trainer

    model = GoNetwork(
        board_size=19
    )

    examples = []

    for _ in range(4):

        state = torch.randn(
            3,
            19,
            19
        ).numpy()

        policy = torch.softmax(
            torch.randn(362),
            dim=0
        ).numpy()

        value = 1.0

        examples.append(
            (
                state,
                policy,
                value
            )
        )

    dataset = SelfPlayDataset(
        examples
    )

    loader = DataLoader(
        dataset,
        batch_size=2
    )

    trainer = Trainer(
        model
    )

    before = (
        model.initial_conv.weight
        .detach()
        .clone()
    )

    trainer.train_epoch(
        loader
    )

    after = (
        model.initial_conv.weight
        .detach()
        .clone()
    )

    assert not torch.equal(
        before,
        after
    )