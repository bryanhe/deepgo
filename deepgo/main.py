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

def main(args=None):

    parser = deepgo.parser()
    args = parser.parse_args(args)

    deepgo.utils.logging.setup_logging(args.logfile, args.loglevel)
    logger = logging.getLogger(__name__)

    ### Log information about run ###
    logger.info(args)
    logger.info("Configuration file: {}".format(deepgo.config.FILENAME))
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    else:
        logger.info("CUDA_VISIBLE_DEVICES not defined.")
    logger.info("CPUs: {}".format(os.sched_getaffinity(0)))
    logger.info("GPUs: {}".format(torch.cuda.device_count()))
    logger.info("Hostname: {}".format(socket.gethostname()))

    ### Seed RNGs ###
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        task = {"prepare": deepgo.prepare,
                "train": deepgo.train}
        task[args.task](args)

    except Exception as e:
        logger.exception(traceback.format_exc())
        raise
