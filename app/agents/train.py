# Following this setup
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

from data_loader import GameData
from torch.utils.data import DataLoader
from loss_fn import SnakeLoss
from agents import TwoLayerAgent
import torch
import datetime
import numpy as np
import torch.nn.utils as utils


# TODO normalize input data coherently in train and infer
# TODO tech model that left when going right does not work


# Set the maximum norm of the gradients
max_norm = 1.0


def train_one_epoch(
    epoch_index, model, training_loader, loss_fn, optimizer, batch_size
):
    running_loss = 0.0
    last_loss = 0.0
    model_par = model.dense1.weight.data[0][0]

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

        # Clip the gradients using the maximum norm
        # utils.clip_grad_norm_(model.parameters(), max_norm)

        # print(np.abs(model.dense1.weight.grad).sum())

        # print(f"{model.dense1.weight.data[0][0]}")
        if model_par != model.dense1.weight.data[0][0]:
            print(f"!!!!!!!!{epoch_index=} {i=}")
        # Adjust learning weights TODO check step is actually taken, because all validations are all the same!!!
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        batch_number = 25
        if (i + 1) % batch_number == 0:
            last_loss = running_loss / (batch_number * batch_size)  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            running_loss = 0.0

    return last_loss


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    batch_size = 256
    data_path = "/home/mas/Github/snake/snake_ai/data/data.csv"
    training_data = GameData(data_path)
    training_loader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_data = GameData(data_path)
    validation_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=1
    )

    loss_fn = SnakeLoss()
    model = TwoLayerAgent(input_dim=9, output_dim=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # optimizer =torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    EPOCHS = 5
    epoch_number = 0

    best_vloss = 1_000_000.0

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        game_state, direction, score = vdata
        voutputs = model(game_state)
        vloss = loss_fn(voutputs, direction, score)
        # print(vloss, game_state.shape)
        running_vloss += vloss.item()

    avg_vloss = running_vloss / ((i + 1) * batch_size)
    print("LOSS valid {}".format(avg_vloss))

    for epoch in range(EPOCHS):
        print("EPOCH {}:".format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            epoch, model, training_loader, loss_fn, optimizer, batch_size
        )

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            game_state, direction, score = vdata
            voutputs = model(game_state)
            vloss = loss_fn(voutputs, direction, score)
            # print(vloss, game_state.shape)
            running_vloss += vloss.item()

        avg_vloss = running_vloss / ((i + 1) * batch_size)
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                 epoch_number + 1)
        # writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = "model_{}_{}".format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
