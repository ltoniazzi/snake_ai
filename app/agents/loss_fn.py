import torch
import torch.nn as nn


class SnakeLoss(nn.Module):
    """
    Sum the the scores weighted by the probabilities assigned to the action chosen"""

    def __init__(self, weihgts: torch.Tensor = 0) -> None:
        super().__init__()
        self.weights = weihgts

    def forward(
        self, direction_pred: torch.Tensor, direction: torch.Tensor, score: torch.Tensor
    ) -> float:
        """Return loss

        Parameters
        ----------
        direction_pred : torch.Tensor
            Shape (B, N, 4)
        direction : torch.Tensor
            Shape (B, N, 1) dtype torch.long
        score : torch.Tensor
            Shape (B, N, 1)

        Returns
        -------
        torch.float
            _description_
        """

        proba_of_chosen_direction = torch.gather(
            direction_pred, dim=-1, index=direction
        )

        predicted_reward = torch.mul(proba_of_chosen_direction, score)

        return torch.sum(predicted_reward)


if __name__ == "__main__":

    # Test loss
    direction_pred = torch.Tensor(
        [
            [
                [0.5, 0.3, 0.2],
                [0.3, 0.5, 0.2],
            ]
        ]
    )

    direction = torch.Tensor(
        [
            [
                [0],
                [2],
            ]
        ],
    ).to(torch.int64)

    score = torch.Tensor(
        [
            [
                [1],
                [2],
            ]
        ]
    )

    loss = SnakeLoss()

    output = loss.forward(direction_pred, direction, score)

    print(output)

    assert output == torch.Tensor([-0.9])
