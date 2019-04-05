import torch
import sgf
import glob
import os
import deepgo
import logging
import numpy as np

class KGS(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        # TODO: actually use split
        with open(os.path.join(deepgo.config.KGS_PROCESSED_ROOT, "info.txt")) as f:
            self.filename, self.move = zip(*map(lambda x: (x.split()[0], int(x.split()[1])), f.readlines()))

        self.len = sum(self.move)
        self.game_index = []
        self.move_index = []
        for (i, m) in enumerate(self.move):
            self.game_index.extend([i] * m)
            self.move_index.extend(range(m))

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        b = []
        # TODO: maybe just stack everything in one npz?
        for move in range(-6, 2):
            move += self.move_index[i]
            if move < 1:
                move = 1
            data = np.load(os.path.join(deepgo.config.KGS_PROCESSED_ROOT, self.filename[self.game_index[i]], "{}.npz".format(move)))
            b.append(data["b"])
        b = np.concatenate(b + [np.full((1, 19, 19), data["p"])])
        b = torch.as_tensor(b, dtype=torch.float)
        m = torch.tensor(data["m"])
        w = torch.tensor(data["w"])
        return b, m, w
