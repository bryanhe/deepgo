import deepgo
import logging
import numpy as np
import torch
import sgf
import pathlib
import glob
import os
import tqdm

def prepare(args=None):
    
    logger = logging.getLogger(__name__)

    ### Select device for computation ###
    device = ("cuda" if args.gpu else "cpu")

    pathlib.Path(deepgo.config.KGS_PROCESSED_ROOT).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(deepgo.config.KGS_PROCESSED_ROOT, "info.txt"), "w") as info:
        for filename in tqdm.tqdm(glob.glob(os.path.join(deepgo.config.KGS_ROOT, "*", "*.sgf"))):
            try:
                with open(filename) as f:
                    f = sgf.parse(f.read())
                    assert(len(f.children) == 1)
                    f = f.children[0]
                    assert(f.children == [])
                    assert(f.nodes != [])
                    node = f.nodes
            except sgf.ParseException:
                logger.warning("SGF parse error encountered.")
                continue

            gamename = os.path.splitext(os.path.basename(filename))[0]
            pathlib.Path(os.path.join(deepgo.config.KGS_PROCESSED_ROOT, gamename)).mkdir(parents=True, exist_ok=True)
            board = torch.zeros((2, 19, 19), device=device, dtype=torch.uint8)
            for (ind, n) in enumerate(node):
                if ind == 0:
                    if "AB" in n.properties:
                        for x in n.properties["AB"]:
                            i, j = deepgo.utils.utils.to_coord(x)
                            board[0, i, j] = 1
                    if "RE" not in n.properties:
                        logger.warning("{} does not have a result.".format(gamename))
                        winner = None
                        break
                    if n.properties["RE"][0][0] == "B":
                        winner = 0
                    elif n.properties["RE"][0][0] == "W":
                        winner = 1
                    else:
                        raise ValueError()
                    if os.path.isfile(os.path.join(deepgo.config.KGS_PROCESSED_ROOT, gamename, "{}.npz".format(len(node) - 1))):
                        break
                else:
                    assert(("B" in n.properties) != ("W" in n.properties))
                    if "B" in n.properties:
                        player = 0
                        move = n.properties["B"]
                    elif "W" in n.properties:
                        player = 1
                        move = n.properties["W"]

                    assert(len(move) == 1)
                    move = move[0]
                    if move == "":
                        # player chose to pass
                        m = 19 * 19
                    else:
                        i, j = deepgo.utils.utils.to_coord(move)
                        board[player, i, j] = 1
                        deepgo.utils.utils.capture(board, 1 - player)  # TODO: assumes that moves won't results in own capture
                        m = i * 19 + j

                    np.savez_compressed(os.path.join(deepgo.config.KGS_PROCESSED_ROOT, gamename, "{}.npz".format(ind)),
                                        b=board.cpu(), p=player, m=m, w=winner)

            if winner is not None:
                info.write("{} {}\n".format(gamename, len(node) - 1))
