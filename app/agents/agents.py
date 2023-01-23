import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerAgent(nn.Module):
    def __init__(
        self, input_dim: int = 3 * 2 + 1 * 2 + 1, output_dim: int = 4, hidden_dim=512
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.dense1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.dense2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input: torch.Tensor):
        """_summary_

        Parameters
        ----------
        input : torch.Tensor
            Shape (1, input_dim)

        Returns
        -------
        _type_
            Shape (1, output_dim)
        """

        input = self.dense1(input)
        input = F.relu(input)
        input = self.dense2(input)
        preds = F.softmax(input, dim=-1)

        return preds

    def infer(self, input):

        output = self.forward(input)

        action, _ = torch.max(output, dim=-1, keepdim=False)

        return action


if __name__ == "__main__":

    # Test agent
    input = torch.Tensor(
        [
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
        ]
    )

    agent = TwoLayerAgent(input_dim=input.shape[-1])

    agent.forward(input)

    print(agent.infer(input))


"""
# pred_proba, pred_proba_index  = torch.max(direction_pred, dim=1, keepdim=True)

# proba_of_chosen_direction = torch.index_select(direction_pred, dim=1, index=direction, out=None)

# broadcast_weights, _ = torch.broadcast_tensors(self.weights, direction_pred)
# weights_indexed = torch.index_select(broadcast_weights, dim=1, index=pred_proba_index, out=None)

# mask = torch.eq(direction_pred, pred_proba.view(-1, 1))
# predicted_proba = torch.mul(pred_proba, mask)
"""
