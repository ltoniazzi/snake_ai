import pandas as pd
from torch.utils.data import Dataset
import torch
import ast


class GameData(Dataset):
    def __init__(self, game_frames, transform=None, target_transform=None):
        self.df = pd.read_csv(game_frames)
        self.score = self.df.Score
        self.game_state = self.df["PlayerX"].apply(lambda x: ast.literal_eval(x))
        self.direction = self.df["Direction"]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        game_state = torch.Tensor(self.game_state.iloc[idx])
        score = torch.Tensor([self.score.iloc[idx]])
        direction = torch.Tensor([self.direction[idx]]).to(torch.int64)
        if self.transform:
            game_state = self.transform(game_state)
        if self.target_transform:
            score = self.target_transform(score)
        return game_state, direction, score


if __name__ == "__main__":

    # df = pd.read_csv()

    data_path = "/home/mas/Github/snake/snake_ai/data/data.csv"

    dl = DataLoader(data_path)

    dl[0]

    print(dl[0])
