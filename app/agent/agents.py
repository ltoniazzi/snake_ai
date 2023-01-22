import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DataLoader(Dataset):

    # import os
    # import pandas as pd
    # from torchvision.io import read_image

    # class CustomImageDataset(Dataset):
    #     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    #         self.img_labels = pd.read_csv(annotations_file)
    #         self.img_dir = img_dir
    #         self.transform = transform
    #         self.target_transform = target_transform

    #     def __len__(self):
    #         return len(self.img_labels)

    #     def __getitem__(self, idx):
    #         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    #         image = read_image(img_path)
    #         label = self.img_labels.iloc[idx, 1]
    #         if self.transform:
    #             image = self.transform(image)
    #         if self.target_transform:
    #             label = self.target_transform(label)
    #         return image, label
    pass


class TwoLayerAgent(nn.Module):

    def __init__(self, input_dim: int = 3*2 + 1*2 + 1, output_dim: int = 4,  hidden_dim = 512 ) -> None:
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


class SnakeLoss(nn.Module):
    """
    Sum the the scores weighted by the probabilities assigned to the action chosen"""

    def __init__(self, weihgts: torch.Tensor = 0) -> None:
        super().__init__()
        self.weights = weihgts


    def forward(self, direction_pred: torch.Tensor, direction: torch.Tensor, score: torch.Tensor) -> float:
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

        proba_of_chosen_direction = torch.gather(direction_pred, dim=2, index=direction)

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

    direction =  torch.Tensor(
        [
                    [
                    [0],
                    [2],
                    ]
        ],
                    ).to(torch.int64)

    score =  torch.Tensor(
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

    assert output == torch.Tensor([0.9])

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