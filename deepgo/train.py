import torch
import torchvision
import numpy as np
import logging
import pathlib
import traceback
import random
import time
import os
import deepgo
import glob
import socket

def train(args=None):

    logger = logging.getLogger(__name__)

    ### Select device for computation ###
    device = ("cuda" if args.gpu else "cpu")

    ### Dataset setup ###
    train_dataset = deepgo.datasets.KGS(split="train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True, pin_memory=args.gpu)

    test_dataset = deepgo.datasets.KGS(split="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True, pin_memory=args.gpu)

    ### Model setup ###
    model = deepgo.models.__dict__[args.model]()

    if args.gpu:
        model = torch.nn.DataParallel(model)
    model.to(device)

    ### Reload parameters from incomplete run ###
    start_epoch = 0
    if args.restart:
        for epoch in range(args.epochs)[::-1]:
            if os.path.isfile(args.checkpoint + str(epoch + 1) + ".pt"):
                start_epoch = epoch + 1
                checkpoint = torch.load(args.checkpoint + str(epoch + 1) + ".pt")
                model.load_state_dict(checkpoint["model"])
                optim.load_state_dict(checkpoint["optim"])
                logger.info("Detected run stopped at epoch #{}. Restarting from checkpoint.".format(start_epoch))
                break

    ### Training Loop ###
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(start_epoch, args.epochs):
        logger.info("Epoch #" + str(epoch + 1))
        for (dataset, loader) in [("train", train_loader), ("test", test_loader)]:

            t = time.time()
            torch.set_grad_enabled(dataset == "train")
            model.train(dataset == "train")

            n = 0
            total = 0
            total_p = 0
            total_y = 0
            correct_p = 0
            correct_y = 0
            logger.info(dataset + ":")
            for (i, (X, p, y)) in enumerate(loader):

                X = X.to(device)
                p = p.to(device)
                y = y.to(device)

                if dataset == "test" and args.average:
                    batch, n_sym, c, h, w = X.shape
                    X = X.view(-1, c, h, w)

                p_hat, y_hat = model(X)

                if dataset == "test" and args.average:
                    pred = pred.view(batch, n_sym, -1).mean(1)

                # Policy loss
                loss_p = torch.nn.functional.cross_entropy(p_hat, p, reduction='sum')
                correct_p += torch.sum(torch.argmax(p_hat, dim=1) == p).cpu().detach().numpy()

                # Value loss
                y_hat = torch.squeeze(y_hat, dim=1)
                loss_y = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y.type(torch.float), reduction='sum')
                correct_y += torch.sum((y_hat > 0) == y.type(torch.uint8)).cpu().detach().numpy()

                loss = loss_p + loss_y
                total += loss.detach().cpu().numpy()
                total_p += loss_p.detach().cpu().numpy()
                total_y += loss_y.detach().cpu().numpy()
                n += X.shape[0]

                message = ""
                message += "{:8d} / {:d} ({:4.0f} / {:4.0f}):".format(i + 1, len(loader), time.time() - t, (time.time() - t) * len(loader) / (i + 1))
                message += "    Loss={:.3f}".format(total / n)
                message += "    Policy={:.3f}".format(total_p / n)
                message += "    Value={:.3f}".format(total_y / n)
                message += "    Policy={:.3f}".format(correct_p / n)
                message += "    Value={:.3f}".format(correct_y / n)
                logger.debug(message)

                if dataset == "train":
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

            if dataset == "train" and args.checkpoint is not None and ((epoch + 1) % args.checkpoint_every) == 0:
                pathlib.Path(os.path.dirname(args.checkpoint)).mkdir(parents=True, exist_ok=True)
                # TODO: if model is on gpu, does loading automatically put on gpu?
                # https://discuss.pytorch.org/t/how-to-store-model-which-trained-by-multi-gpus-dataparallel/6526
                # torch.save(model.state_dict(), args.checkpoint + str(epoch + 1) + ".pt")
                torch.save({
                    'model': model.state_dict(),
                    'optim' : optim.state_dict(),
                }, args.checkpoint + str(epoch + 1) + ".pt")

                if epoch != 0 and not args.keep_all_checkpoints:
                    os.remove(args.checkpoint + str(epoch + 1 - args.checkpoint_every) + ".pt")
