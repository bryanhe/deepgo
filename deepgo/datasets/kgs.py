import torch
import sgf
import glob
import os
import deepgo
import logging
import numpy as np

class KGS(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        with open(os.path.join(deepgo.config.KGS_PROCESSED_ROOT, "info.txt")) as f:
            self.filename, self.move = zip(*map(lambda x: (x.split()[0], int(x.split()[1])), f.readlines()))

        # TODO: shuffle games?
        n_games = len(self.filename)
        if split == "train":
            self.filename = self.filename[:round(0.8 * n_games)]
            self.move = self.move[:round(0.8 * n_games)]
        elif split == "test":
            self.filename = self.filename[round(0.8 * n_games):]
            self.move = self.move[round(0.8 * n_games):]
        else:
            raise ValueError()


        self.len = sum(self.move)
        self.game_index = []
        self.move_index = []
        for (i, m) in enumerate(self.move):
            self.game_index.extend([i] * m)
            self.move_index.extend(range(m))

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        data = np.load(os.path.join(deepgo.config.KGS_PROCESSED_ROOT, "{}.npz".format(self.filename[self.game_index[i]])))
        ind = self.move_index[i] - np.array(range(8))
        ind[ind < 0] = 0
        b = torch.as_tensor(np.concatenate([data["b"][ind, :, :, :].reshape(16, 19, 19)] + [np.full((1, 19, 19), data["p"][self.move_index[i]])]), dtype=torch.float)
        m = torch.tensor(data["m"][self.move_index[i]])
        w = torch.tensor(data["w"])
        return b, m, w
