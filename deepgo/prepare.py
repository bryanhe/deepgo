import deepgo
import logging
import numpy as np
import torch
import sgf
import pathlib
import glob
import os
import tqdm
import time

def prepare(args=None):
    
    logger = logging.getLogger(__name__)

    ### Select device for computation ###
    device = ("cuda" if args.gpu else "cpu")

    pathlib.Path(deepgo.config.KGS_PROCESSED_ROOT).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(deepgo.config.KGS_PROCESSED_ROOT, "info.txt"), "w") as info:
        for filename in tqdm.tqdm(glob.glob(os.path.join(deepgo.config.KGS_ROOT, "*", "*.sgf"))):
            t = time.time()
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
            npz_filename = os.path.join(deepgo.config.KGS_PROCESSED_ROOT, "{}.npz".format(gamename))

            board = []
            player = []
            move = []
            b = torch.zeros((2, 19, 19), device=device, dtype=torch.uint8)
            for (ind, n) in enumerate(node):
                if ind == 0:
                    if "AB" in n.properties:
                        for x in n.properties["AB"]:
                            i, j = deepgo.utils.utils.to_coord(x)
                            b[0, i, j] = 1
                    if "RE" not in n.properties:
                        logger.warning("{} does not have a result.".format(gamename))
                        winner = None
                        break
                    if n.properties["RE"][0][0] == "B":
                        winner = 0
                    elif n.properties["RE"][0][0] == "W":
                        winner = 1
                    else:
                        logger.warning("{} has result \"{}\".".format(gamename, n.properties["RE"][0]))
                        winner = None
                        break
                    if os.path.isfile(npz_filename):
                        break
                else:
                    assert(("B" in n.properties) != ("W" in n.properties))
                    if "B" in n.properties:
                        p = 0
                        m = n.properties["B"]
                    elif "W" in n.properties:
                        p = 1
                        m = n.properties["W"]
                    player.append(p)

                    assert(len(m) == 1)
                    m = m[0]
                    if m == "":
                        # player chose to pass
                        m = 19 * 19
                    else:
                        i, j = deepgo.utils.utils.to_coord(m)
                        b[p, i, j] = 1
                        deepgo.utils.utils.capture(b, 1 - p)  # TODO: assumes that moves won't results in own capture
                        m = i * 19 + j

                    move.append(m)
                    board.append(b.unsqueeze(0).cpu().numpy().copy())

            if winner is not None:
                if not os.path.isfile(npz_filename) and len(node) > 1:
                    np.savez_compressed(npz_filename,
                                        b=np.concatenate(board), p=np.array(player), m=np.array(move), w=winner)
                info.write("{} {}\n".format(gamename, len(node) - 1))
