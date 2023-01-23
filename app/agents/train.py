# Following this setup
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

from data_loader import GameData
from torch.utils.data import DataLoader
from loss_fn import SnakeLoss
from agents import TwoLayerAgent
import torch

epochs = 5

data_path = "/home/mas/Github/snake/snake_ai/data/data.csv"
training_data = GameData(data_path)
training_loader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=1)
loss_fn = SnakeLoss()
model = TwoLayerAgent(input_dim=3, output_dim=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


def train_one_epoch(epoch_index):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        game_state, direction, score = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(game_state)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, direction, score)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 != 999:
            last_loss = running_loss / 1000  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            running_loss = 0.0

    return last_loss


if __name__ == "__main__":

    train_one_epoch(1)
