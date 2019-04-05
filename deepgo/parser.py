def parser():
    """Returns parser."""

    import argparse
    import argcomplete
    from datetime import datetime as dt
    from . import __version__
    from .config import config

    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    parser.add_argument("--seed", "-s", type=int, default=0, help="RNG seed")

    subparsers = parser.add_subparsers(dest="task", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare dataset")
    add_device_arguments(prepare, default=False)

    train = subparsers.add_parser("train", help="run training")
    add_model_arguments(train)
    add_device_arguments(train)
    add_training_arguments(train)
    train.add_argument("--average", action="store_true", help="average between rotations and reflections")

    argcomplete.autocomplete(parser)

    return parser


def add_model_arguments(parser):
    parser.add_argument("--model", "-m", default="dual39",
                        # choices=sorted(name for name in deepgo.models.__dict__ if name.islower() and not name.startswith("__") and callable(deepgo.models.__dict__[name])),  TODO: autocomplete speed issue
                        help="model architecture")

    parser.add_argument("--load", type=str, default=None, help="weights to load")

    parser.add_argument("--checkpoint_every", type=int, default=1, help="how frequently to save checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="root for checkpoints")
    parser.add_argument("--restart", action="store_true", help="automatically reload checkpoint")
    parser.add_argument("--keep_all_checkpoints", action="store_true", help="do not delete old checkpoints")


def add_device_arguments(parser, default=True):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpu", action="store_const", const=True, dest="gpu", default=default, help="use GPU")
    group.add_argument("--cpu", action="store_const", const=False, dest="gpu", default=default, help="use CPU")


def add_training_arguments(parser):
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay for SGD")

    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch", type=int, default=64, help="batch size")

    parser.add_argument("--workers", type=int, default=4, help="number of workers for dataloader")


def add_logging_arguments(parser):
    import logging

    def loglevel(level):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError("Invalid log level: %s" % loglevel)
        return numeric_level

    parser.add_argument("--loglevel", "-l", type=loglevel,
                        default=logging.DEBUG, help="logging level")
    parser.add_argument("--logfile", type=str,
                        default=None,
                        help="file to store logs")
