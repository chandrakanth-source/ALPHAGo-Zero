import torch
import torch.nn.functional as F


class Trainer:

    def __init__(
        self,
        model,
        learning_rate=0.001,
        weight_decay=1e-4,
        device=None
    ):

        self.model = model

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    def calculate_loss(
        self,
        predicted_policy,
        predicted_value,
        target_policy,
        target_value
    ):

        policy_loss = -torch.mean(
            torch.sum(
                target_policy
                * F.log_softmax(
                    predicted_policy,
                    dim=1
                ),
                dim=1
            )
        )

        value_loss = F.mse_loss(
            predicted_value.squeeze(-1),
            target_value
        )

        total_loss = (
            policy_loss
            + value_loss
        )

        return (
            total_loss,
            policy_loss,
            value_loss
        )
    def train_epoch(
        self,
        dataloader
    ):

        self.model.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        batches = 0

        for states, policies, values in dataloader:

            states = states.to(self.device)
            policies = policies.to(self.device)
            values = values.to(self.device)

            self.optimizer.zero_grad()

            predicted_policy, predicted_value = (
                self.model(states)
            )

            (
                loss,
                policy_loss,
                value_loss
            ) = self.calculate_loss(
                predicted_policy,
                predicted_value,
                policies,
                values
            )

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

            batches += 1

        if batches == 0:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0
            }

        return {
            "loss": total_loss / batches,
            "policy_loss": total_policy_loss / batches,
            "value_loss": total_value_loss / batches
        }
    def save_checkpoint(
        self,
        path,
        epoch
    ):

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict()
        }

        torch.save(
            checkpoint,
            path
        )
    def load_checkpoint(
        self,
        path
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        return checkpoint["epoch"]